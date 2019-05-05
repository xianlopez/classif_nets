# ======================================================================================================================
import tensorflow as tf
import tools
import logging
import os
import cv2
import numpy as np
import sys
import DataAugmentation
import Resizer
from models import model_factory
from PIL import Image


# VGG_MEAN = [123.68, 116.78, 103.94]
VGG_MEAN = [123.0, 117.0, 104.0]


# ======================================================================================================================
class InteractiveDataReader:

    def __init__(self, input_width, input_height, args):
        self.input_width = input_width
        self.input_height = input_height
        self.preprocess_type = args.preprocess_opts.type
        self.mean = args.preprocess_opts.mean
        self.range_min = args.preprocess_opts.range_min
        self.range_max = args.preprocess_opts.range_max
        self.resize_method = args.resize_method
        self.img_extension, self.classnames = tools.process_dataset_config(args.dataset_info_path)
        if args.preprocess_opts.mean == 'vgg':
            self.mean = [123.68, 116.78, 103.94]
        else:
            raise Exception('Preprocess mean not recognized.')

    def build_inputs(self):
        return tf.placeholder(dtype=tf.float32, shape=(None, self.input_width, self.input_height, 3), name='inputs')

    def get_batch(self, image_paths):
        batch_size = len(image_paths)
        inputs_numpy = np.zeros(shape=(batch_size, self.input_width, self.input_height, 3), dtype=np.float32)
        for i in range(batch_size):
            inputs_numpy[i, :, :, :] = self.get_image(image_paths[i])
        return inputs_numpy

    def preprocess_image(self, image):
        # Subtract mean or fit to range:
        if self.preprocess_type == 'subtract_mean':
            means = np.zeros(shape=image.shape, dtype=np.float32)
            for i in range(3):
                means[:, :, i] = self.mean[i]
            image = image - means
        elif self.preprocess_type == 'fit_to_range':
            image = self.range_min + image * (self.range_max - self.range_min) / 255.0
        else:
            raise Exception('Preprocess type not recognized.')
        # Resize:
        image = Resizer.ResizeNumpy(image, self.resize_method, self.input_width, self.input_height)
        return image

    def get_image(self, image_path):
        # Read image:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Preprocess it:
        image = self.preprocess_image(image)
        return image


# ======================================================================================================================
class InteractiveTamperingDataReader(InteractiveDataReader):
    def build_inputs(self):
        inputs_left = tf.placeholder(dtype=tf.float32, shape=(None, self.input_width, self.input_height, 3), name='inputs_left')
        inputs_right = tf.placeholder(dtype=tf.float32, shape=(None, self.input_width, self.input_height, 3), name='inputs_right')
        return inputs_left, inputs_right

    def get_batch(self, image_paths):
        if len(image_paths) != 2:
            raise Exception('Expected a list of two elements as input.')
        image_paths_left = image_paths[0]
        image_paths_right = image_paths[1]
        batch_size = len(image_paths_left)
        if len(image_paths_left) != len(image_paths_right):
            raise Exception('Different number of image paths for left and right inputs.')
        inputs_left_numpy = np.zeros(shape=(batch_size, self.input_width, self.input_height, 3), dtype=np.float32)
        inputs_right_numpy = np.zeros(shape=(batch_size, self.input_width, self.input_height, 3), dtype=np.float32)
        for i in range(batch_size):
            inputs_left_numpy[i, :, :, :] = self.get_image(image_paths_left[i])
            inputs_right_numpy[i, :, :, :] = self.get_image(image_paths_right[i])
        return inputs_left_numpy, inputs_right_numpy


# ======================================================================================================================
class TrainDataReader:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shape, args):

        self.batch_size = args.batch_size
        self.input_width = input_shape[0]
        self.input_height = input_shape[1]
        self.num_workers = args.num_workers
        self.buffer_size = args.buffer_size

        self.resize_function = Resizer.ResizerSimple(self.input_width, self.input_height).get_resize_func(args.resize_method)

        self.percent_of_data = args.percent_of_data
        self.max_image_size = args.max_image_size
        self.nimages_train = None
        self.nimages_val = None
        self.train_init_op = None
        self.val_init_op = None
        self.dirdata = os.path.join(args.root_of_datasets, args.dataset_name)
        self.img_extension, self.classnames = tools.process_dataset_config(os.path.join(self.dirdata, 'dataset_info.xml'))
        self.img_extension = '.' + self.img_extension
        self.nclasses = len(self.classnames)
        self.outdir = args.outdir
        self.write_network_input = args.write_network_input

        self.shuffle_data = args.shuffle_data

        if self.img_extension == '.jpg' or self.img_extension == '.JPEG':
            self.parse_function = parse_jpg
        elif self.img_extension == '.png':
            self.parse_function = parse_png
        else:
            raise Exception('Images format not recognized.')

        self.data_aug_opts = args.data_aug_opts

        if self.data_aug_opts.apply_data_augmentation:
            bugs_class_id = -1
            for i in range(len(self.classnames)):
                if self.classnames[i] == 'BUGS':
                    bugs_class_id = i
                    break
            data_augmenter = DataAugmentation.ClassificationDataAugmentation(args, self.input_width, self.input_height, bugs_class_id)
            self.data_aug_func = data_augmenter.data_augmenter
        return

    # ------------------------------------------------------------------------------------------------------------------
    def get_nbatches_per_epoch(self, split):

        if split == 'train':
            return self.nimages_train / self.batch_size
        elif split == 'val':
            return self.nimages_val / self.batch_size
        else:
            raise Exception('Split not recognized.')

    # ------------------------------------------------------------------------------------------------------------------
    def get_init_op(self, split):

        if split == 'train':
            return self.train_init_op
        elif split == 'val':
            return self.val_init_op
        else:
            raise Exception('Split not recognized.')

    # ------------------------------------------------------------------------------------------------------------------
    def build_iterator(self):

        self.read_count = 0

        batched_dataset_train, self.nimages_train = self.build_batched_dataset('train')
        print('Number of training examples: ' + str(self.nimages_train))
        batched_dataset_val, self.nimages_val = self.build_batched_dataset('val')
        print('Number of validation examples: ' + str(self.nimages_val))

        iterator = tf.data.Iterator.from_structure(batched_dataset_train.output_types,
                                                    batched_dataset_train.output_shapes)

        inputs, labels, filenames = iterator.get_next(name='iterator-output')
        self.train_init_op = iterator.make_initializer(batched_dataset_train, name='train_init_op')
        self.val_init_op = iterator.make_initializer(batched_dataset_val, name='val_init_op')

        return inputs, labels, filenames

    # ------------------------------------------------------------------------------------------------------------------
    def build_batched_dataset(self, split):

        labels_file = os.path.join(self.dirdata, split + '_labels.txt')
        filenames, labels = self.read_paths_and_labels(labels_file)
        batched_dataset = self.build_classification_dataset(filenames, labels, split)

        return batched_dataset, len(filenames)

    # ------------------------------------------------------------------------------------------------------------------
    def build_classification_dataset(self, filenames, labels, split):

        # This dataset receives as input a 1-D tensor with the names of all the image files, and another 1-D tensor
        # with their associated labels (no one-hot encoding, just the class index).
        # The parse function simply reads the image.
        # The preprocess is then applied.
        # Finally, the dataset outputs the preprocessed images, the (unmodified) labels, and the filenames.
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(self.parse_classif_w_all, num_parallel_calls=self.num_workers)

        if split == 'train' and self.data_aug_opts.apply_data_augmentation:
            dataset = dataset.map(self.data_aug_func, num_parallel_calls=self.num_workers)

        dataset = dataset.map(self.preprocess_w_all, num_parallel_calls=self.num_workers)
        dataset = dataset.map(self.resize_func_extended_classification, num_parallel_calls=self.num_workers)
        if self.shuffle_data:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        return dataset.batch(self.batch_size)


    # ------------------------------------------------------------------------------------------------------------------
    def resize_func_extended_classification(self, image, label, filename):
        image = self.resize_function(image)
        return image, label, filename


    # ------------------------------------------------------------------------------------------------------------------
    def preprocess_w_all(self, image, label, filename):
        means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
        image = image - means
        return image, label, filename


    # ------------------------------------------------------------------------------------------------------------------
    def parse_classif_w_all(self, filename, label):
        image = self.parse_function(filename)
        return image, label, filename


    # ------------------------------------------------------------------------------------------------------------------
    def read_paths_and_labels(self, labels_file):

        paths = []
        labels = []

        try:
            with open(labels_file, 'r') as file:
                for line in file:
                    line_split = line.split(',')
                    paths.append(os.path.join(self.dirdata, tools.adapt_path_to_current_os(line_split[0])))
                    labels.append(int(line_split[1]))
        except FileNotFoundError as ex:
            logging.error('File ' + labels_file + ' does not exist.')
            logging.error(str(ex))
            raise

        # Remove data or shuffle:
        if self.percent_of_data != 100:
            # Remove data:
            indexes = np.random.choice(np.arange(len(labels)), int(self.percent_of_data / 100.0 * len(labels)), replace=False)
        else:
            # Shuffle data at least:
            indexes = np.arange(len(labels))
            if self.shuffle_data:
                np.random.shuffle(indexes)

        aux_paths = paths
        aux_labels = labels
        paths = []
        labels = []

        for i in range(len(indexes)):
            paths.append(aux_paths[indexes[i]])
            labels.append(aux_labels[indexes[i]])

        # Remove the remaining examples that do not fit in a batch.
        if len(paths) % self.batch_size != 0:

            aux_paths = paths
            aux_labels = labels
            paths = []
            labels = []

            for i in range(len(aux_paths) - (len(aux_paths) % self.batch_size)):
                paths.append(aux_paths[i])
                labels.append(aux_labels[i])

        assert len(paths) % self.batch_size == 0, 'Number of images is not a multiple of batch size'

        return paths, labels


    def write_network_input_pyfunc(self, image, bboxes):
        img = image.copy()
        height = img.shape[0]
        width = img.shape[1]
        min_val = np.min(img)
        img = img - min_val
        max_val = np.max(img)
        img = img / float(max_val) * 255.0
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for box in bboxes:
            class_id = int(box[0])
            xmin = int(np.round(box[1] * width))
            ymin = int(np.round(box[2] * height))
            w = int(np.round(box[3] * width))
            h = int(np.round(box[4] * height))
            cv2.rectangle(img, (xmin, ymin), (xmin + w, ymin + h), (0, 0, 255), 2)
            cv2.rectangle(img, (xmin, ymin - 20),
                          (xmin + w, ymin), (125, 125, 125), -1)
            cv2.putText(img, self.classnames[class_id], (xmin + 5, ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        number = 0
        file_path_candidate = os.path.join(self.outdir, 'input' + str(number) + '.png')
        while os.path.exists(file_path_candidate):
            number += 1
            file_path_candidate = os.path.join(self.outdir, 'input' + str(number) + '.png')
        cv2.imwrite(file_path_candidate, img)
        return image


    def write_network_input_func(self, image, bboxes, filename):
        shape = image.shape
        image = tf.py_func(self.write_network_input_pyfunc, [image, bboxes], tf.float32)
        image.set_shape(shape)
        return image, bboxes, filename


# ----------------------------------------------------------------------------------------------------------------------
def parse_jpg(filepath):
    img = tf.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)

    return img


# ----------------------------------------------------------------------------------------------------------------------
def parse_png(filepath):

    img = tf.read_file(filepath)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32)

    return img


# ----------------------------------------------------------------------------------------------------------------------
# This is done to avoid memory problems.
def ensure_max_size(image, max_size):

    img_height, img_width, _ = image.shape
    factor = np.sqrt(max_size * max_size / (img_height * img_width))

    if factor < 1:
        new_width = int(img_width * factor)
        new_height = int(img_height * factor)
        image = cv2.resize(image, (new_width, new_height))
    else:
        factor = 1

    return image, factor
