import tensorflow as tf
import time
import numpy as np
import os
import tools

num_workers = 8
buffer_size = 200
VGG_MEAN = [123.0, 117.0, 104.0]
input_height = 224
input_width = 448
batch_size = 16
n_steps = 10


def parse_func(filename):
        image = tf.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        return image


def preprocess_func(image):
        means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
        image = image - means
        return image


def resize_func(image):
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    scale_height = input_height / tf.to_float(height)
    scale_width = input_width / tf.to_float(width)
    scale = tf.minimum(scale_height, scale_width)
    size = tf.cast(tf.stack([scale*tf.to_float(height), scale*tf.to_float(width)]), tf.int32)
    image = tf.image.resize_images(image, size)
    image = tf.image.resize_image_with_crop_or_pad(image, input_height, input_width)
    return image


def build_dataset(filenames):
    dataset = tf.data.Dataset.from_tensor_slices((filenames))
    dataset = dataset.map(parse_func, num_parallel_calls=num_workers)
    dataset = dataset.map(preprocess_func, num_parallel_calls=num_workers)
    dataset = dataset.map(resize_func, num_parallel_calls=num_workers)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    return dataset


def read_paths_and_labels(labels_file, dirdata):
    paths = []
    labels = []
    try:
        with open(labels_file, 'r') as file:
            for line in file:
                line_split = line.split(',')
                paths.append(os.path.join(dirdata, tools.adapt_path_to_current_os(line_split[0])))
                labels.append(int(line_split[1]))
    except FileNotFoundError as ex:
        print('File ' + labels_file + ' does not exist.')
        print(str(ex))
        raise
    # Shuffle data:
    indexes = np.arange(len(labels))
    np.random.shuffle(indexes)
    aux_paths = paths
    aux_labels = labels
    paths = []
    labels = []
    for i in range(len(indexes)):
        paths.append(aux_paths[indexes[i]])
        labels.append(aux_labels[indexes[i]])
    # Remove the remaining examples that do not fit in a batch.
    if len(paths) % batch_size != 0:
        aux_paths = paths
        aux_labels = labels
        paths = []
        labels = []
        for i in range(len(aux_paths) - (len(aux_paths) % batch_size)):
            paths.append(aux_paths[i])
            labels.append(aux_labels[i])
    assert len(paths) % batch_size == 0, 'Number of images is not a multiple of batch size'
    return paths, labels


def speed_test_5():
    dataset_dir = os.path.join(os.path.dirname(tools.get_base_dir()), 'datasets', 'coco-animals')
    labels_file = os.path.join(dataset_dir, 'train_labels.txt')
    filenames, labels = read_paths_and_labels(labels_file, dataset_dir)
    batched_dataset = build_dataset(filenames)
    iterator = tf.data.Iterator.from_structure(batched_dataset.output_types, batched_dataset.output_shapes)
    x = iterator.get_next(name='iterator-output')
    train_init_op = iterator.make_initializer(batched_dataset, name='train_init_op')

    net = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.sigmoid)
    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.sigmoid)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    current_shape = net.shape
    # net = tf.reshape(net, [batch_size, -1])
    net = tf.reshape(net, [-1, current_shape[1] * current_shape[2] * current_shape[3]])
    net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=net, units=10)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(train_init_op)
        for i in range(n_steps):
            x_np = np.random.normal(size=(batch_size, input_height, input_width, 3))
            ini = time.time()
            sess.run(fetches=[logits], feed_dict={x: x_np})
            fin = time.time()
            print('Step ' + str(i) + ' done in ' + str(fin - ini) + ' s.')


if __name__ == '__main__':
    speed_test_5()
