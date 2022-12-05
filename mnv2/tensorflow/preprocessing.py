"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import tensorflow as tf


def resize_image(
        image: tf.Tensor,
        height: int,
        width: int) -> tf.Tensor:
    """
    Resizes an image to a given height and width.

    :param image: `Tensor` representing an image of arbitrary size.
    :param height: image height.
    :param width: image width.
    :return: a float32 tensor containing the resized image.
    """
    return tf.compat.v1.image.resize(
        image,
        [height, width],
        method=tf.image.ResizeMethod.BILINEAR,
        align_corners=False)


def center_crop(image: tf.Tensor,
                image_size: int,
                crop_padding: int) -> tf.Tensor:
    """
    Crops to center of image with padding then scales image_size.

    :param image: `Tensor` representing an image of arbitrary size.
    :param image_size: image height/width dimension.
    :param crop_padding: the padding size to use when centering the crop.
    :return: a cropped image `Tensor`.
    """
    shape = tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        ((image_size / (image_size + crop_padding)) *
         tf.cast(tf.minimum(image_height, image_width), tf.float32)),
        tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2

    image = tf.image.crop_to_bounding_box(
        image,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=padded_center_crop_size,
        target_width=padded_center_crop_size)

    image = resize_image(image=image,
                         height=image_size,
                         width=image_size)

    return image


def crop_and_flip(image: tf.Tensor) -> tf.Tensor:
    """
    Crops an image to a random part of the image, then randomly flips.

    :param image: `Tensor` representing an image of arbitrary size.
    :return: a cropped image `Tensor`.
    """
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    shape = tf.shape(image)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Reassemble the bounding box in the format the crop op requires.
    offset_height, offset_width, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    cropped = tf.image.crop_to_bounding_box(
        image,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=target_height,
        target_width=target_width)

    # Flip to add a little more random distortion in.
    cropped = tf.image.random_flip_left_right(cropped)
    return cropped
