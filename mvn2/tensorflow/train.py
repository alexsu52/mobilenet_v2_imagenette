import argparse

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from preprocessing import center_crop
from preprocessing import crop_and_flip
from preprocessing import resize_image

parser = argparse.ArgumentParser(description='TensorFlow Imagenette Training')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

DATASET_CLASSES = 10


def main():
    args = parser.parse_args()

    # choose strategy
    strategy = tf.distribute.get_strategy()

    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 1:
        strategy =  tf.distribute.MirroredStrategy()

    # Data loading code
    train_dataset, info = tfds.load('imagenette/320px-v2', split='train',
                                    shuffle_files=True, as_supervised=True, 
                                    with_info=True)

    replica_batch_size = args.batch_size // strategy.num_replicas_in_sync
    global_batch_size = replica_batch_size * strategy.num_replicas_in_sync
    steps_per_epoch = info.splits['train'].num_examples // global_batch_size

    def preprocess_for_train(image, label):
        image = crop_and_flip(image)
        image = resize_image(image, 224, 224)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = tf.image.convert_image_dtype(image, tf.float32)

        label = tf.one_hot(label, DATASET_CLASSES)

        return image, label

    train_dataset = (train_dataset.map(preprocess_for_train,
                                       num_parallel_calls=tf.data.AUTOTUNE)
                                  .batch(replica_batch_size,
                                         drop_remainder=True)
                                  .prefetch(tf.data.AUTOTUNE))

    validation_dataset = tfds.load('imagenette/320px-v2', split='validation', 
                                   shuffle_files=False, as_supervised=True)
    
    def preprocess_for_eval(image, label):
        image = center_crop(image, 224, 32)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = tf.image.convert_image_dtype(image, tf.float32)

        label = tf.one_hot(label, DATASET_CLASSES)

        return image, label

    validation_dataset = (validation_dataset.map(preprocess_for_eval, 
                                                num_parallel_calls=tf.data.AUTOTUNE)
                                            .batch(replica_batch_size,
                                                  drop_remainder=False)
                                            .prefetch(tf.data.AUTOTUNE))

    with strategy.scope():
        # create the model
        weights = None
        if args.pretrained:
            weights = 'imagenet'
        imagenet_model = tf.keras.applications.MobileNetV2(weights=weights)
        
        input = imagenet_model.input
        output = imagenet_model.layers[-2].output
        x = tf.keras.layers.Dense(DATASET_CLASSES, activation='softmax', 
                                  name='predictions')(output)
        model = tf.keras.Model(input, x, name='mobilenetv2_imagenette')
        model.summary()

        # define loss function (criterion), optimizer, and learning rate scheduler
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.1,
            patience=10,
            min_lr=1e-6,
            verbose=1)

        optimizer = tf.keras.optimizers.SGD(
            learning_rate = args.lr,
            momentum = args.momentum,
            weight_decay = args.weight_decay)


        loss_obj = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

        metrics = [
             tf.keras.metrics.CategoricalAccuracy(name='acc@1'),
             tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='acc@5'),
             tfa.metrics.MeanMetricWrapper(loss_obj, name='loss'),
        ]

        model.compile(optimizer=optimizer,
                      loss=loss_obj,
                      metrics=metrics)

        initial_epoch = 0
        if args.resume:
            model.load_weights(args.resume)
            initial_epoch = model.optimizer.iterations.numpy() // steps_per_epoch

    callbacks = [
        reduce_lr,
        tf.keras.callbacks.TensorBoard(),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss', 
            patience=15, 
            verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='checkpoints',
            monitor='val_acc@1',
            save_best_only=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='tf_model.h5',
            monitor='val_acc@1',
            save_best_only=True, 
            save_weights_only=True,
            verbose=1)
    ]

    if args.evaluate:
        model.evaluate(validation_dataset, verbose=1).expect_partial()
        return

    model.fit(
        train_dataset,
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        validation_data=validation_dataset)


if __name__ == '__main__':
    main()
