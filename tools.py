# ======================================================================================================================
import time
import os
import logging
import sys
from shutil import copyfile
import numpy as np
import cv2
from lxml import etree
import tensorflow as tf
import scipy.misc
import matplotlib
matplotlib.use('Agg')  # To avoid exception 'async handler deleted by the wrong thread'
from matplotlib import pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
base_dir = None
def get_base_dir():
    global base_dir
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return base_dir
    else:
        return base_dir


# ----------------------------------------------------------------------------------------------------------------------
def adapt_path_to_current_os(path):
    if os.sep == '\\': # Windows
        path = path.replace('/', os.sep)
    else: # Linux
        path = path.replace('\\', os.sep)
    return path



# ----------------------------------------------------------------------------------------------------------------------
def process_dataset_config(dataset_info_path):

    dataset_config_file = os.path.join(dataset_info_path)
    tree = etree.parse(dataset_config_file)
    root = tree.getroot()
    images_format = root.find('format').text
    classes = root.find('classes')
    classnodes = classes.findall('class')
    classnames = [''] * len(classnodes)

    for cn in classnodes:
        classid = cn.find('id').text
        name = cn.find('name').text
        assert classid.isdigit(), 'Class id must be a non-negative integer.'
        assert int(classid) < len(classnodes), 'Class id greater or equal than classes number.'
        classnames[int(classid)] = name

    for i in range(len(classnames)):
        assert classnames[i] != '', 'Name not found for id ' + str(i)

    return images_format, classnames


# ----------------------------------------------------------------------------------------------------------------------
def plot_training_history(train_metrics, train_loss, val_metrics, val_loss, args, epoch_num):

    if len(train_loss) >= 2 or len(val_loss) >= 2:

        # Epochs on which we computed train and validation measures:
        x_train = np.arange(args.nepochs_checktrain, epoch_num + 1, args.nepochs_checktrain)
        x_val = np.arange(args.nepochs_checkval, epoch_num + 1, args.nepochs_checkval)
        # Initialize figure:
        # Axis 1 will be for metrics, and axis 2 for losses.
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        if len(train_loss) >= 2:
            # Train loss:
            ax2.plot(x_train, train_loss, 'b-', label='train loss')
            # Train metric:
            ax1.plot(x_train, train_metrics, 'r-', label='train accuracy')
        if len(val_loss) >= 2:
            # Val loss:
            ax2.plot(x_val, val_loss, 'b--', label='val loss')
            # Val metric:
            ax1.plot(x_val, val_metrics, 'r--', label='val accuracy')

        # Axis limits for metrics:
        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, np.max(np.concatenate((train_loss, val_loss))))

        # Add title
        plt.title('Train history')

        # Add axis labels
        ax1.set_ylabel('Accuracy')
        ax2.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')

        # To adjust correctly everything:
        fig.tight_layout()

        # Add legend
        ax1.legend(loc='upper left')
        ax2.legend(loc='lower left')

        # Delete previous figure to save the new one
        fig_path = os.path.join(args.outdir, 'train_history.png')
        if os.path.exists(fig_path):
            try:
                os.remove(fig_path)
            except:
                logging.warning('Error removing ' + fig_path + '. Using alternative name.')
                fig_path = os.path.join(args.outdir, 'train_history_' + str(epoch_num) + '.png')

        # Save fig
        plt.savefig(fig_path)

        # Close plot
        plt.close()

    return


# ----------------------------------------------------------------------------------------------------------------------
def write_results(predictions, labels, filepaths, classnames, args):

    if args.write_images or args.write_results:
        logging.info('Writing results...')
        nimages = len(predictions)

        if args.write_results:
            with open(os.path.join(args.outdir, 'results.txt'), 'w') as fid:
                for i in range(nimages):
                    predicted_class = np.argmax(np.array(predictions[i]))
                    imgpath = filepaths[i].decode(sys.getdefaultencoding())
                    path, filename = os.path.split(imgpath)
                    fid.write(filename + ' - Predicted: ' + classnames[predicted_class] + '\n')
            # Confusion matrix:
            if labels is not None:
                conf_matrix = np.zeros((len(classnames), len(classnames)), dtype=np.int32)
                for i in range(len(predictions)):
                    predicted_class = np.argmax(np.array(predictions[i]))
                    real_class = labels[i]
                    conf_matrix[real_class, predicted_class] += 1
                with open(os.path.join(args.outdir, 'confusion_matrix.csv'), 'w') as fid:
                    line = 'Label \\ Prediction'
                    for i in range(len(classnames)):
                        line += ';' + classnames[i]
                    line += '\n'
                    fid.write(line)
                    for i in range(len(classnames)):
                        line = classnames[i]
                        for j in range(len(classnames)):
                            line += ';' + str(conf_matrix[i, j])
                        line += '\n'
                        fid.write(line)

        if args.write_images:

            imgsdir = os.path.join(args.outdir, 'images')
            os.makedirs(imgsdir)

            for cl_name in classnames:
                os.makedirs(os.path.join(imgsdir, cl_name))

            for i in range(nimages):
                predicted_class = np.argmax(np.array(predictions[i]))
                imgpath = filepaths[i].decode(sys.getdefaultencoding())
                _, filename = os.path.split(imgpath)
                pathout = os.path.join(imgsdir, classnames[predicted_class], filename)
                pathout = ensure_new_path(pathout)
                img = cv2.imread(imgpath)
                cv2.imwrite(pathout, img)
        logging.info('Results written.')

    return



# ----------------------------------------------------------------------------------------------------------------------
def create_experiment_folder(args):
    year = time.strftime('%Y')
    month = time.strftime('%m')
    day = time.strftime('%d')
    if not os.path.exists(args.experiments_folder):
        os.mkdir(args.experiments_folder)
    year_folder = os.path.join(args.experiments_folder, year)
    if not os.path.exists(year_folder):
        os.mkdir(year_folder)
    base_name = os.path.join(year_folder, year + '_' + month + '_' + day)
    experiment_folder = base_name
    count = 0
    while os.path.exists(experiment_folder):
        count += 1
        experiment_folder = base_name + '_' + str(count)
    os.mkdir(experiment_folder)
    print('Experiment folder: ' + experiment_folder)
    return experiment_folder


# ----------------------------------------------------------------------------------------------------------------------
def copy_config(args, inline_args):
    if inline_args.run == 'train':
        configModuleName = 'train_config'
    elif inline_args.run == 'evaluate':
        configModuleName = 'eval_config'
    elif inline_args.run == 'predict':
        configModuleName = 'predict_config'
    else:
        print('Please, select specify a valid execution mode: train / evaluate / predict')
        raise Exception()

    if inline_args.conf is not None:
        configModuleName = configModuleName + '_' + inline_args.conf
        configModuleNameAndPath = os.path.join('config', configModuleName)
    else:
        configModuleNameAndPath = configModuleName

    configModuleNameAndPath = os.path.join(get_base_dir(), configModuleNameAndPath + '.py')

    copyfile(configModuleNameAndPath, os.path.join(args.outdir, configModuleName + '.py'))
    return


# ----------------------------------------------------------------------------------------------------------------------
def configure_logging(args):
    if len(logging.getLogger('').handlers) == 0:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=os.path.join(args.outdir, 'out.log'),
                            filemode='w')
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler(sys.stdout)
        # console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
    else:
        file_handler = None
        for handler in logging.getLogger('').handlers:
            if type(handler) == logging.FileHandler:
                file_handler = handler
        if file_handler is None:
            raise Exception('File handler not found.')
        logging.getLogger('').removeHandler(file_handler)
        fileh = logging.FileHandler(filename=os.path.join(args.outdir, 'out.log'), mode='w')
        formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
        fileh.setFormatter(formatter)
        fileh.setLevel(logging.DEBUG)
        logging.getLogger('').addHandler(fileh)
    logging.info('Logging configured.')


# ----------------------------------------------------------------------------------------------------------------------
def get_config_proto(gpu_memory_fraction):
    if gpu_memory_fraction > 0:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    else:
        gpu_options = tf.GPUOptions()
    return tf.ConfigProto(gpu_options=gpu_options)




# ----------------------------------------------------------------------------------------------------------------------
def save_input_images(names, images, args, epoch_num, batch_num, img_extension, labels=None):
    input_imgs_dir = os.path.join(args.outdir, 'input_images')
    if not os.path.exists(input_imgs_dir):
        os.makedirs(input_imgs_dir)
    epoch_dir = os.path.join(input_imgs_dir, 'epoch_'+str(epoch_num))
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)
    batch_dir = os.path.join(epoch_dir, 'batch_'+str(batch_num))
    if not os.path.exists(batch_dir):
        os.makedirs(batch_dir)
    for i in range(len(names)):
        filename = names[i].decode(sys.getdefaultencoding())
        # Remove drive:
        if ':' in filename:
            _, filename = os.path.split(filename)
        # Split subfolders:
        subfolders = filename.split(os.sep)
        img_folder = batch_dir
        for j in range(len(subfolders) - 1):
            img_folder = os.path.join(img_folder, subfolders[j])
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
        # Final name and path:
        filename = subfolders[len(subfolders) - 1]
        img_path = os.path.join(img_folder, filename)
        # print(img_path)
        if img_path[len(img_path)-4:] != img_extension:
            img_path = img_path + img_extension
        image_to_write = images[i]
        max_value = np.max(image_to_write)
        if labels is not None:
            for box in labels[i]:
                x0 = int(box.x0)
                y0 = int(box.y0)
                x1 = min(x0 + int(box.width), image_to_write.shape[1]-1)
                y1 = min(y0 + int(box.height), image_to_write.shape[0]-1)
                image_to_write[y0, x0:x1, 0] = max_value
                image_to_write[y1, x0:x1, 0] = max_value
                image_to_write[y0:y1, x0, 0] = max_value
                image_to_write[y0:y1, x1, 0] = max_value
        scipy.misc.imsave(img_path, image_to_write)
    return



# ----------------------------------------------------------------------------------------------------------------------
def ensure_new_path(path_in):
    dot_pos = path_in.rfind('.')
    rawname = path_in[:dot_pos]
    extension = path_in[dot_pos:]
    new_path = path_in
    count = 0
    while os.path.exists(new_path):
        count += 1
        new_path = rawname + '_' + str(count) + extension
    return new_path


# ----------------------------------------------------------------------------------------------------------------------
def get_trainable_variables(args):
    # Choose the variables to train:
    if args.layers_list is None or args.layers_list == []:
        # Train all variables
        vars_to_train = tf.trainable_variables()

    else:
        # Train the variables included in layers_list
        if args.train_selected_layers:

            vars_to_train = []
            for v in tf.trainable_variables():
                selected = False
                for layer in args.layers_list:
                    if layer in v.name:
                        selected = True
                        break
                if selected:
                    vars_to_train.append(v)

        # Train the variables NOT included in layers_list
        else:

            vars_to_train = []
            for v in tf.trainable_variables():
                selected = True
                for layer in args.layers_list:
                    if layer in v.name:
                        selected = False
                        break
                if selected:
                    vars_to_train.append(v)
    return vars_to_train
