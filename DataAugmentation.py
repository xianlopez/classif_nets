import tensorflow as tf
import os
import cv2
import tools
import numpy as np
import sys


####### DATA AUGMENTATION ########
class DataAugOpts:
    apply_data_augmentation = False  # If false, none of the following options have any effect.
    horizontal_flip = False
    vertical_flip = False
    random_brightness = False
    brightness_prob = 0.5
    brightness_delta_lower = -32
    brightness_delta_upper = 32
    random_contrast = False
    contrast_prob = 0.5
    contrast_factor_lower = 0.5
    contrast_factor_upper = 1.5
    random_saturation = False
    saturation_prob = 0.5
    saturation_factor_lower = 0.5
    saturation_factor_upper = 1.5
    random_hue = False
    hue_prob = 0.5
    hue_delta_lower = -0.1
    hue_delta_upper = 0.1
    rtss_rc = False  # Resize To Smallest Side and Random Crop
    smallest_side = 256
    random_crop = False  # This options and rtss_rc cannot be chosen at the same time.
    random_crop_proportion = 0.7
    random_crop_prob = 0.2
    rotation_degree = 0
    rotation_prob = 0.5
    convert_to_grayscale_prob = 0
##################################


class ClassificationDataAugmentation:
    def __init__(self, args, input_width, input_height, bugs_class_id=-1):
        self.input_width = input_width
        self.input_height = input_height
        self.data_aug_opts = args.data_aug_opts
        self.outdir = args.outdir
        self.write_image_after_data_augmentation = args.write_image_after_data_augmentation
        self.bugs_class_id = bugs_class_id
        if args.num_workers > 1 and args.write_image_after_data_augmentation:
            raise Exception('Option write_image_after_data_augmentation is not compatible with more than one worker to load data')

    def data_augmenter(self, image, label, filename):
        if self.data_aug_opts.horizontal_flip:
            image = tf.image.random_flip_left_right(image)
        if self.data_aug_opts.vertical_flip:
            image = tf.image.random_flip_up_down(image)
        if self.data_aug_opts.random_brightness:
            image = random_adjust_brightness(image, self.data_aug_opts.brightness_delta_lower,
                                             self.data_aug_opts.brightness_delta_upper,
                                             self.data_aug_opts.brightness_prob)
        if self.data_aug_opts.random_contrast:
            image = random_adjust_contrast(image, self.data_aug_opts.contrast_factor_lower,
                                           self.data_aug_opts.contrast_factor_upper,
                                           self.data_aug_opts.contrast_prob)
        if self.data_aug_opts.random_saturation:
            image = random_adjust_saturation(image, self.data_aug_opts.saturation_factor_lower,
                                             self.data_aug_opts.saturation_factor_upper,
                                             self.data_aug_opts.saturation_prob)
        if self.data_aug_opts.random_hue:
            image = random_adjust_hue(image, self.data_aug_opts.hue_delta_lower,
                                      self.data_aug_opts.hue_delta_upper,
                                      self.data_aug_opts.hue_prob)
        if self.data_aug_opts.convert_to_grayscale_prob > 0:
            image = convert_to_grayscale(image, self.data_aug_opts.convert_to_grayscale_prob)
        if self.data_aug_opts.rotation_degree != 0:
            image = self.rotate(image, self.data_aug_opts.rotation_prob)
        if self.data_aug_opts.rtss_rc:
            image = self.rtss_rc(image)
        elif self.data_aug_opts.random_crop:
            image = random_crop(image, self.data_aug_opts.random_crop_proportion, self.data_aug_opts.random_crop_prob)
        if self.write_image_after_data_augmentation:
            image = tf.py_func(self.write_image, [image, filename], tf.float32)
            image.set_shape((None, None, 3))
        return image, label, filename

    def rotate(self, image, prob):
        flag = tf.random_uniform(()) < prob
        radians = self.data_aug_opts.rotation_degree / 360.0 * 2.0 * np.pi
        angle = tf.random_uniform(shape=(), minval=-radians, maxval=radians)
        image_rotated = tf.contrib.image.rotate(image, angle, interpolation='BILINEAR')
        image = tf.cond(flag, lambda: image_rotated, lambda: image)
        return image

    # Resize To Smallest Side and Random Crop
    def rtss_rc(self, image):
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        height = tf.to_float(height)
        width = tf.to_float(width)
        scale = tf.cond(tf.greater(height, width),
                        lambda: self.data_aug_opts.smallest_side / width,
                        lambda: self.data_aug_opts.smallest_side / height)
        new_height = tf.to_int32(height * scale)
        new_width = tf.to_int32(width * scale)
        image = tf.image.resize_images(image, [new_height, new_width])
        image = tf.random_crop(image, [self.input_width, self.input_height, 3])
        return image

    def write_image(self, image, file_path):
        file_path_str = file_path.decode(sys.getdefaultencoding())
        file_name = os.path.basename(file_path_str)
        raw_name = os.path.splitext(file_name)[0]
        file_path_candidate = os.path.join(self.outdir, 'image_after_data_aug_' + raw_name + '.png')
        file_path = tools.ensure_new_path(file_path_candidate)
        print('path to save image: ' + file_path)
        # print(str(np.min(image)) + '   ' + str(np.mean(image)) + '   ' + str(np.max(image)))
        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, img)
        return image


def random_crop(image, min_proportion, prob):
    proportion = tf.random_uniform(shape=(), minval=min_proportion, maxval=1)
    # proportion = tf.Print(proportion, [proportion], 'proportion')
    proportion_vec = tf.stack([proportion, proportion, tf.ones(shape=(), dtype=tf.float32)], axis=0)
    original_size = tf.shape(image)
    # original_size = tf.Print(original_size, [original_size], 'original_size')
    new_size = tf.cast(tf.round(tf.cast(original_size, tf.float32) * proportion_vec), tf.int32)
    # new_size = tf.Print(new_size, [new_size], 'new_size')
    crop = tf.random_crop(image, new_size)
    flag = tf.random_uniform(()) < prob
    image = tf.cond(flag, lambda: crop, lambda: tf.identity(image))
    return image


def adjust_contrast(image, factor):
    image = tf.clip_by_value(127.5 + factor * (image - 127.5), 0, 255)
    return image

def random_adjust_contrast(image, factor_lower, factor_upper, prob):
    factor = tf.random_uniform(shape=(), minval=factor_lower, maxval=factor_upper)
    flag = tf.random_uniform(()) < prob
    image = tf.cond(flag, lambda: adjust_contrast(image, factor), lambda: image)
    return image

def adjust_brightness(image, brightness_delta):
    image = tf.clip_by_value(tf.image.adjust_brightness(image, brightness_delta), 0, 255)
    return image

def random_adjust_brightness(image, delta_lower, delta_upper, prob):
    delta_brightness = tf.random_uniform(shape=(), minval=delta_lower, maxval=delta_upper)
    flag = tf.random_uniform(()) < prob
    image = tf.cond(flag, lambda: adjust_brightness(image, delta_brightness), lambda: image)
    return image

def random_adjust_saturation(image, factor_lower, factor_upper, prob):
    factor = tf.random_uniform(shape=(), minval=factor_lower, maxval=factor_upper)
    flag = tf.random_uniform(()) < prob
    image = tf.cond(flag, lambda: tf.image.adjust_saturation(image, factor), lambda: image)
    return image

def random_adjust_hue(image, delta_lower, delta_upper, prob):
    delta_hue = tf.random_uniform(shape=(), minval=delta_lower, maxval=delta_upper)
    flag = tf.random_uniform(()) < prob
    image = tf.cond(flag, lambda: tf.image.adjust_hue(image, delta_hue), lambda: image)
    return image

def convert_to_grayscale(image, prob):
    image_gray = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])
    flag = tf.random_uniform(()) < prob
    image = tf.cond(flag, lambda: image_gray, lambda: image)
    return image