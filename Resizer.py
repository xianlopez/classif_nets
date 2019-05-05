import tensorflow as tf
import cv2
import numpy as np


class ResizerSimple:
    def __init__(self, input_width, input_height):
        self.input_width = input_width
        self.input_height = input_height

    def get_resize_func(self, resize_method):
        if resize_method == 'resize_warp':
            return self.resize_warp
        elif resize_method == 'resize_pad_zeros':
            return self.resize_pad_zeros
        elif resize_method == 'resize_lose_part':
            return self.resize_lose_part
        elif resize_method == 'centered_crop':
            return self.centered_crop
        else:
            raise Exception('Resize method not recognized.')

    def resize_warp(self, image):
        image = tf.image.resize_images(image, [self.input_height, self.input_width])
        return image

    def resize_pad_zeros(self, image):
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        scale_height = self.input_height / tf.to_float(height)
        scale_width = self.input_width / tf.to_float(width)
        scale = tf.minimum(scale_height, scale_width)
        size = tf.cast(tf.stack([scale*tf.to_float(height), scale*tf.to_float(width)]), tf.int32)
        image = tf.image.resize_images(image, size)
        image = tf.image.resize_image_with_crop_or_pad(image, self.input_height, self.input_width)
        return image

    def resize_lose_part(self, image):
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        scale_height = self.input_height / tf.to_float(height)
        scale_width = self.input_width / tf.to_float(width)
        scale = tf.maximum(scale_height, scale_width)
        size = tf.cast(tf.stack([scale*tf.to_float(height), scale*tf.to_float(width)]), tf.int32)
        image = tf.image.resize_images(image, size)
        image = tf.image.resize_image_with_crop_or_pad(image, self.input_height, self.input_width)
        return image

    def centered_crop(self, image):
        image = tf.image.resize_image_with_crop_or_pad(image, self.input_height, self.input_width)
        return image


def ResizeNumpy(image, method, input_width, input_height):
    if method == 'resize_warp':
        image = cv2.resize(image, (input_width, input_height))
    elif method == 'resize_pad_zeros':
        height, width, _ = image.shape
        # Resize so it fits totally in the input size:
        scale_width = input_width / np.float32(width)
        scale_height = input_height / np.float32(height)
        scale = min(scale_width, scale_height)
        new_width = int(np.round(width * scale))
        new_height = int(np.round(height * scale))
        image = cv2.resize(image, (new_width, new_height))
        # Pad with zeros the remaining areas:
        increment_height = int(input_height - new_height)
        increment_top = int(np.round(increment_height / 2.0))
        increment_bottom = increment_height - increment_top
        increment_width = int(input_width - new_width)
        increment_left = int(np.round(increment_width / 2.0))
        increment_right = increment_width - increment_left
        image = cv2.copyMakeBorder(image, increment_top, increment_bottom, increment_left, increment_right,
                                   cv2.BORDER_CONSTANT)
    elif method == 'resize_lose_part':
        raise Exception('resize_lose_part not implemented for InteractiveDataReader')
    elif method == 'centered_crop':
        raise Exception('centered_crop not implemented for InteractiveDataReader')
    else:
        raise Exception('Resize method not recognized.')
    return image

