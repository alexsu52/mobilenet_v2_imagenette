import argparse
import os
import shutil
import time
from enum import Enum

from torch.optim.lr_scheduler import ReduceLROnPlateau
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras.metrics import Metric

from tensorflow.preprocessing import center_crop, crop_and_flip, resize_image

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='FOOD101',
                    help='path to dataset (default: dataset)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
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
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

best_acc1 = 0

DATASET_DIR = '~/.cache/nncf/datasets'
DATASET_CLASSES = 10


# def get_config_from_argv(argv, parser):
#     args = parser.parse_args(args=argv)
#     config = create_sample_config(args, parser)
#     configure_paths(config)
#     return config


# def get_dataset_builders(config, num_devices, one_hot=True):
#     image_size = config.input_info.sample_size[-2]

#     train_builder = DatasetBuilder(
#         config,
#         image_size=image_size,
#         num_devices=num_devices,
#         one_hot=one_hot,
#         is_train=True)

#     val_builder = DatasetBuilder(
#         config,
#         image_size=image_size,
#         num_devices=num_devices,
#         one_hot=one_hot,
#         is_train=False)

#     return train_builder, val_builder


# def get_num_classes(dataset):
#     if 'imagenet2012' in dataset:
#         num_classes = 1000
#     elif dataset == 'cifar100':
#         num_classes = 100
#     elif dataset == 'cifar10':
#         num_classes = 10
#     else:
#         num_classes = 1000

#     logger.info('The sample is started with {} classes'.format(num_classes))
#     return num_classes


# def load_checkpoint(checkpoint, ckpt_path):
#     logger.info('Load from checkpoint is enabled.')
#     if tf.io.gfile.isdir(ckpt_path):
#         path_to_checkpoint = tf.train.latest_checkpoint(ckpt_path)
#         logger.info('Latest checkpoint: {}'.format(path_to_checkpoint))
#     else:
#         path_to_checkpoint = ckpt_path if tf.io.gfile.exists(ckpt_path + '.index') else None
#         logger.info('Provided checkpoint: {}'.format(path_to_checkpoint))

#     if not path_to_checkpoint:
#         logger.info('No checkpoint detected.')
#         if ckpt_path:
#             raise RuntimeError(f'ckpt_path was given, but no checkpoint detected in path: {ckpt_path}')

#     logger.info('Checkpoint file {} found and restoring from checkpoint'
#                 .format(path_to_checkpoint))

#     status = checkpoint.restore(path_to_checkpoint)
#     status.expect_partial()
#     logger.info('Completed loading from checkpoint.')


# def resume_from_checkpoint(checkpoint, ckpt_path, steps_per_epoch):
#     load_checkpoint(checkpoint, ckpt_path)
#     initial_step = checkpoint.model.optimizer.iterations.numpy()
#     initial_epoch = initial_step // steps_per_epoch

#     logger.info('Resuming from epoch %d', initial_epoch)
#     return initial_epoch


# def load_compression_state(ckpt_path: str):
#     checkpoint = tf.train.Checkpoint(compression_state=TFCompressionStateLoader())
#     load_checkpoint(checkpoint, ckpt_path)
#     return checkpoint.compression_state.state

def main():
    args = parser.parse_args()

    # choose strategy
    strategy = tf.distribute.get_strategy()

    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 1:
        strategy =  tf.distribute.MirroredStrategy()

    # Data loading code
    replica_batch_size = args.batch_size // strategy.num_replicas_in_sync
    global_batch_size = replica_batch_size * strategy.num_replicas_in_sync

    train_dataset = tfds.load('imagenette/320px-v2', split='train',
                              shuffle_files=True, as_supervised=True)

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
        images = center_crop(image, 224, 32)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        images = tf.image.convert_image_dtype(images, tf.float32)

        label = tf.one_hot(label, DATASET_CLASSES)

        return images, label

    validation_dataset = (validation_dataset.map(preprocess_for_eval, 
                                                num_parallel_calls=tf.data.AUTOTUNE)
                                            .batch(replica_batch_size,
                                                  drop_remainder=False)
                                            .prefetch(tf.data.AUTOTUNE))


    with strategy.scope():
        from tensorflow.python.keras.
        model = tf.keras.applications.MobileNetV2()

            scheduler = build_scheduler(
                config=config,
                steps_per_epoch=train_steps)
            optimizer = build_optimizer(
                config=config,
                scheduler=scheduler)

            loss_obj = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

            compress_model.add_loss(compression_ctrl.loss)

            metrics = [
                tf.keras.metrics.CategoricalAccuracy(name='acc@1'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='acc@5'),
                tfa.metrics.MeanMetricWrapper(loss_obj, name='ce_loss'),
                tfa.metrics.MeanMetricWrapper(compression_ctrl.loss, name='cr_loss')
            ]

            compress_model.compile(optimizer=optimizer,
                                   loss=loss_obj,
                                   metrics=metrics,
                                   run_eagerly=config.get('eager_mode', False))

            compress_model.summary()

            checkpoint = tf.train.Checkpoint(model=compress_model,
                                             compression_state=TFCompressionState(compression_ctrl))

            initial_epoch = 0
            if resume_training:
                initial_epoch = resume_from_checkpoint(checkpoint=checkpoint,
                                                       ckpt_path=config.ckpt_path,
                                                       steps_per_epoch=train_steps)

    callbacks = get_callbacks(
        include_tensorboard=True,
        track_lr=True,
        profile_batch=0,
        initial_step=initial_epoch * train_steps,
        log_dir=config.log_dir,
        ckpt_dir=config.checkpoint_save_dir,
        checkpoint=checkpoint)

    callbacks.append(get_progress_bar(
        stateful_metrics=['loss'] + [metric.name for metric in metrics]))
    callbacks.extend(compression_callbacks)

    validation_kwargs = {
        'validation_data': validation_dataset,
        'validation_steps': validation_steps,
        'validation_freq': config.test_every_n_epochs,
    }

    if 'train' in config.mode:
        if is_accuracy_aware_training(config):
            logger.info('starting an accuracy-aware training loop...')
            result_dict_to_val_metric_fn = lambda results: 100 * results['acc@1']
            compress_model.accuracy_aware_fit(train_dataset,
                                              compression_ctrl,
                                              nncf_config=config.nncf_config,
                                              callbacks=callbacks,
                                              initial_epoch=initial_epoch,
                                              steps_per_epoch=train_steps,
                                              tensorboard_writer=SummaryWriter(config.log_dir,
                                                                               'accuracy_aware_training'),
                                              log_dir=config.log_dir,
                                              uncompressed_model_accuracy=uncompressed_model_accuracy,
                                              result_dict_to_val_metric_fn=result_dict_to_val_metric_fn,
                                              **validation_kwargs)
        else:
            logger.info('training...')
            compress_model.fit(
                train_dataset,
                epochs=train_epochs,
                steps_per_epoch=train_steps,
                initial_epoch=initial_epoch,
                callbacks=callbacks,
                **validation_kwargs)

    logger.info('evaluation...')
    statistics = compression_ctrl.statistics()
    logger.info(statistics.to_str())
    results = compress_model.evaluate(
        validation_dataset,
        steps=validation_steps,
        callbacks=[get_progress_bar(
            stateful_metrics=['loss'] + [metric.name for metric in metrics])],
        verbose=1)

    if config.metrics_dump is not None:
        write_metrics(results[1], config.metrics_dump)

    if 'export' in config.mode:
        save_path, save_format = get_saving_parameters(config)
        compression_ctrl.export_model(save_path, save_format)
        logger.info('Saved to {}'.format(save_path))

    close_strategy_threadpool(strategy)


if __name__ == '__main__':
    main()
