import argparse

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from preprocessing import center_crop

parser = argparse.ArgumentParser(description='TensorFlow Imagenette Evaluation')
parser.add_argument('checkpoint', type=str, metavar='PATH', 
                    help='path to the checkpoint')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

DATASET_CLASSES = 10


def main():
    args = parser.parse_args()

    # choose strategy
    strategy = tf.distribute.get_strategy()

    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 1:
        strategy =  tf.distribute.MirroredStrategy()

    # Data loading code
    replica_batch_size = args.batch_size // strategy.num_replicas_in_sync

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
        imagenet_model = tf.keras.applications.MobileNetV2()
        
        input = imagenet_model.input
        output = imagenet_model.layers[-2].output
        x = tf.keras.layers.Dense(DATASET_CLASSES, activation='softmax', 
                                  name='predictions')(output)
        model = tf.keras.Model(input, x, name='mobilenetv2_imagenette')
        model.summary()

        # define loss function (criterion), optimizer, and learning rate scheduler
        loss_obj = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

        metrics = [
             tf.keras.metrics.CategoricalAccuracy(name='acc@1'),
             tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='acc@5'),
             tfa.metrics.MeanMetricWrapper(loss_obj, name='loss'),
        ]

        model.compile(loss=loss_obj,
                      metrics=metrics)

        if args.checkpoint.endswith('.h5'):
            model.load_weights(args.checkpoint)
        else:
            model.load_weights(args.checkpoint).expect_partial()
        
    model.evaluate(validation_dataset, verbose=1)

if __name__ == '__main__':
    main()
