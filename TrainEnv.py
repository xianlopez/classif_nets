import DataReader
import tensorflow as tf
import tools
import logging
from models import model_factory
import time
import numpy as np
import operator
import webbrowser
import subprocess
import socket
import os
from L2Regularization import L2RegularizationLoss
import losses
import math
import re
from LRScheduler import LRScheduler
import sys
import accuracy


class Checkpoint:
    def __init__(self, path, val_loss, val_acc=-1):
        self.path = path
        self.val_loss = val_loss
        self.val_acc = val_acc


# ======================================================================================================================
class TrainEnv:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, args, exec_mode):

        self.loss = None
        self.optimizer = None
        self.net_output = None
        self.predictions = None
        self.inputs = None
        self.labels = None
        self.filenames = None
        self.train_op = None
        self.saver = None
        self.init_op = None
        self.restore_fn = None
        self.classnames = None
        self.nclasses = None
        self.input_shape = None
        self.model_variables = None
        self.reader = None
        self.action = exec_mode  # 'train', 'evaluate'
        self.is_training = None

        self.nbatches_accum = args.nbatches_accum
        self.zero_ops = None
        self.accum_ops = None
        self.train_step = None

        # Initialize network:
        self.generate_graph(args)

    # ------------------------------------------------------------------------------------------------------------------
    def start_interactive_session(self, args):
        logging.info('Starting interactive session')
        self.sess = tf.Session(config=tools.get_config_proto(args.gpu_memory_fraction))
        self.initialize(self.sess, args)
        logging.info('Interactive session started')
        return self.sess

    def forward_batch(self, batch, args):
        predictions = self.sess.run(fetches=self.predictions, feed_dict={self.is_training: False, self.inputs: batch})
        return predictions

    # ------------------------------------------------------------------------------------------------------------------
    def evaluate(self, args, split):

        logging.info("Start evaluation")
        with tf.Session(config=tools.get_config_proto(args.gpu_memory_fraction)) as sess:

            assert type(self.saver) == tf.train.Saver, 'Saver is not correctly initialized'
            # Initialization:
            self.initialize(sess, args)
            # Process all data:
            logging.info('Computing metrics on ' + split + ' data')
            initime = time.time()
            sess.run(self.reader.get_init_op(split))
            nbatches = self.reader.get_nbatches_per_epoch(split)
            step = 0
            all_predictions = []
            all_labels = []
            all_names = []
            while True:
                try:
                    predictions, labels, names, images = sess.run([self.predictions, self.labels, self.filenames, self.inputs], {self.is_training: False})
                    all_predictions.extend(predictions)
                    all_labels.extend(labels)
                    all_names.extend(names)

                    if args.save_input_images:
                        tools.save_input_images(names, images, args, 1, step, self.reader.img_extension)
                    step += 1

                except tf.errors.OutOfRangeError:
                    break

                if step % args.nsteps_display == 0:
                    print('Step %i / %i' % (step, nbatches))

            metrics = accuracy.compute_accuracy(all_predictions, all_labels)
            fintime = time.time()
            logging.debug('Done in %.2f s' % (fintime - initime))
            logging.info(split + ' accuracy: %.2f' % metrics)

            # Write results:
            tools.write_results(all_predictions, all_labels, all_names, self.classnames, args)

        return metrics

    # ------------------------------------------------------------------------------------------------------------------
    def evaluate_on_dataset(self, split, sess, args):

        logging.info('Computing loss and metrics on ' + split + ' data')
        initime = time.time()
        sess.run(self.reader.get_init_op(split))
        nbatches = self.reader.get_nbatches_per_epoch(split)
        step = 0
        all_predictions = []
        all_labels = []
        all_names = []
        loss_acum = 0

        while True:
            try:
                predictions, labels, loss, names = sess.run(
                    [self.predictions, self.labels, self.loss, self.filenames],
                    {self.is_training: False})
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                all_names.extend(names)
                loss_acum += loss
                step += 1

            except tf.errors.OutOfRangeError:
                break

            if step % args.nsteps_display == 0:
                print('Step %i / %i, loss: %.2e' % (step, nbatches, loss))

        metrics = accuracy.compute_accuracy(all_predictions, all_labels)
        loss_per_image = loss_acum / nbatches
        fintime = time.time()
        logging.debug('Done in %.2f s' % (fintime - initime))

        return loss_per_image, metrics, all_predictions, all_labels, all_names

    # ------------------------------------------------------------------------------------------------------------------
    def train(self, args):

        print('')
        logging.info("Start training")
        nbatches_train = self.reader.get_nbatches_per_epoch('train')
        lr_scheduler = LRScheduler(args.lr_scheduler_opts, args.outdir)
        with tf.Session(config=tools.get_config_proto(args.gpu_memory_fraction)) as sess:

            assert type(self.saver) == tf.train.Saver, 'Saver is not correctly initialized'
            # Initialization:
            self.initialize(sess, args)
            # Lists for the training history:
            train_metrics = []
            train_losses = []
            val_metrics = []
            val_losses = []
            best_val_metric = 0
            checkpoints = []  # This is a list of Checkpoint objects.

            # Tensorboard:
            merged, summary_writer, tensorboard_url = prepare_tensorboard(sess, args.outdir)

            # Loop on epochs:
            current_lr = args.learning_rate
            global_step = 0
            for epoch in range(1, args.num_epochs + 1):
                print('')
                logging.info('Starting epoch %d / %d' % (epoch, args.num_epochs))
                sess.run(self.reader.get_init_op('train'))
                current_lr = lr_scheduler.GetLearningRateAtEpoch(epoch, current_lr)
                _ = sess.run([self.update_lr_op], feed_dict={self.lr_to_update: current_lr})
                learning_rate = sess.run([self.learning_rate])[0]
                logging.info('Learning rate: ' + str(learning_rate))
                step = 0
                all_predictions = []
                all_labels = []
                loss_acum = 0
                iniepoch = time.time()

                while True:
                    try:
                        if self.nbatches_accum > 0:
                            if step % self.nbatches_accum == 0:
                                sess.run(self.zero_ops)
                            _, loss, predictions, labels, summaryOut = sess.run([self.accum_ops, self.loss, self.predictions, self.labels, merged], {self.is_training: True})
                            if (step + 1) % self.nbatches_accum == 0:
                                _ = sess.run([self.train_step])

                        else:

                            ini = time.time()
                            _, loss, predictions, labels, summaryOut = sess.run([self.train_op, self.loss, self.predictions, self.labels, merged], {self.is_training: True})
                            fin = time.time()
                            print('Step ' + str(step) + ' done in ' + str(fin - ini) + ' s.')

                        if math.isnan(loss):
                            raise Exception("Loss is Not A Number")

                        if epoch % args.nepochs_checktrain == 0 and not args.recompute_train:
                            all_predictions.extend(predictions)
                            all_labels.extend(labels)
                            loss_acum += loss

                        step += 1
                        global_step += 1

                        # Tensorboard:
                        summary_writer.add_summary(summaryOut, global_step)
                        if global_step == 1:
                            webbrowser.open_new_tab(tensorboard_url)

                    except tf.errors.OutOfRangeError:
                        break

                    if step % args.nsteps_display == 0:
                        logging.info('Step %i / %i, loss: %.2e' % (step, nbatches_train, loss))

                finepoch = time.time()
                logging.debug('Epoch computed in %.2f s' % (finepoch - iniepoch))

                # Compute loss and metrics on training data:
                if epoch % args.nepochs_checktrain == 0:
                    if args.recompute_train:
                        train_loss, metrics, _, _, _ = self.evaluate_on_dataset('train', sess, args)
                        train_losses.append(train_loss)
                        train_metrics.append(metrics)
                        logging.info('Train loss: %.2e' % train_loss)
                        logging.info('Train accuracy: %.2f' % metrics)
                    else:
                        train_loss = loss_acum / nbatches_train
                        train_losses.append(train_loss)
                        logging.info('Mean train loss during epoch: %.2e' % train_loss)
                        metrics = accuracy.compute_accuracy(all_predictions, all_labels)
                        train_metrics.append(metrics)
                        logging.info('Mean train accuracy during epoch: %.2f' % metrics)
                else:
                    train_loss = None

                # Compute loss and metrics on validation data:
                if epoch % args.nepochs_checkval == 0:
                    val_loss, metrics, _, _, _ = self.evaluate_on_dataset('val', sess, args)
                    val_losses.append(val_loss)
                    val_metrics.append(metrics)
                    logging.info('Val loss: %.2e' % val_loss)
                    logging.info('Val accuracy: %.2f' % metrics)
                else:
                    val_loss = None

                # Plot training progress:
                if epoch % args.nepochs_checktrain == 0 or epoch % args.nepochs_checkval == 0:
                    tools.plot_training_history(train_metrics, train_losses, val_metrics, val_losses, args, epoch)

                # Save the model:
                if epoch % args.nepochs_save == 0:
                    # save_path = self.saver.save(sess, os.path.join(args.outdir, 'model'), global_step=epoch)
                    # logging.info('Model saved to ' + save_path)
                    self.save_checkpoint_classif(sess, val_loss, val_metrics[0], epoch, checkpoints, args.outdir)

            # Save the model (if we haven't done it yet):
            if args.num_epochs % args.nepochs_save != 0:
                # save_path = self.saver.save(sess, os.path.join(args.outdir, 'model'), global_step=args.num_epochs)
                # logging.info('Model saved to ' + save_path)
                self.save_checkpoint_classif(sess, val_loss, val_metrics[0], epoch, checkpoints, args.outdir)

            best_val_metric = np.max(np.array(val_metrics, dtype=np.float32))
            print('Best validation metric: ' + str(best_val_metric))

        return best_val_metric

    def save_checkpoint_classif(self, sess, val_loss, val_acc, epoch, checkpoints, outdir):
        if val_loss is None:
            val_loss = -1
        if val_acc is None:
            val_acc = -1
        # Save new model:
        save_path = self.saver.save(sess, os.path.join(outdir, 'model'), global_step=epoch)
        logging.info('Model saved to ' + save_path)
        new_checkpoint = Checkpoint(save_path, val_loss, val_acc)

        if len(checkpoints) > 0:
            # Remove all the previous checkpoints but the best one.
            checkpoints.sort(key=operator.attrgetter('val_acc'), reverse=True)
            for i in range(len(checkpoints) - 2, -1, -1):
                # Remove:
                ckpt = checkpoints[i]
                folder, name = os.path.split(ckpt.path)
                for file in os.listdir(folder):
                    if re.search(name, file) is not None:
                        file_path = os.path.join(folder, file)
                        try:
                            os.remove(file_path)
                        except Exception as ex:
                            logging.warning('Error deleting file ' + file_path, exc_info=ex)
                checkpoints.pop(i)
                logging.info('Deleted checkpoint. Path: ' + ckpt.path + '  -  Val accuracy: ' + str(ckpt.val_acc))
            # If the remaining checkpoint is worse than the new checkpoint, remove it too.
            ckpt = checkpoints[0]
            if ckpt.val_acc <= val_acc:
                # Remove:
                folder, name = os.path.split(ckpt.path)
                for file in os.listdir(folder):
                    if re.search(name, file) is not None:
                        file_path = os.path.join(folder, file)
                        try:
                            os.remove(file_path)
                        except Exception as ex:
                            logging.warning('Error deleting file ' + file_path, exc_info=ex)
                checkpoints.pop(0)
                logging.info('Deleted checkpoint. Path: ' + ckpt.path + '  -  Val accuracy: ' + str(ckpt.val_acc))

        # Append the new checkpoint to the list:
        checkpoints.append(new_checkpoint)

        logging.info('Remaining checkpoints:')
        for ckpt in checkpoints:
            logging.info('Path: ' + ckpt.path + '  -  Val accuracy: ' + str(ckpt.val_acc))
        return checkpoints

    # ------------------------------------------------------------------------------------------------------------------
    def generate_graph(self, args):
        self.define_inputs_and_labels(args)
        self.build_model(args)

        if self.action == 'train':
            self.build_loss(args)
            self.build_optimizer(args)

        self.define_initializer(args)
        self.saver = tf.train.Saver(name='net_saver', max_to_keep=1000000)

    # ------------------------------------------------------------------------------------------------------------------
    def define_inputs_and_labels(self, args):

        self.input_shape = model_factory.define_input_shape(args)

        if self.action == 'interactive':
            self.reader = DataReader.InteractiveDataReader(self.input_shape[0], self.input_shape[1], args)
            self.inputs = self.reader.build_inputs()
        else:
            self.reader = DataReader.TrainDataReader(self.input_shape, args)
            self.inputs, self.labels, self.filenames = self.reader.build_iterator()
        self.classnames = self.reader.classnames
        self.nclasses = len(self.classnames)

    # ------------------------------------------------------------------------------------------------------------------
    def build_model(self, args):
        self.net_output, self.predictions, self.model_variables, self.is_training = model_factory.build_model(self.inputs, self.nclasses, args)

    # ------------------------------------------------------------------------------------------------------------------
    def build_loss(self, args):

        self.loss = losses.cross_entropy(self.labels, self.net_output)

        if args.l2_regularization > 0:
            self.loss += L2RegularizationLoss(args)

        self.loss = tf.identity(self.loss, name='loss') # This is just a workaround to rename the loss function to 'loss'
        # self.loss = tf.divide(self.loss, np.float32(args.batch_size), name='loss')

        # self.loss = tf.Print(self.loss, [self.loss], 'total loss')

        # Tensorboard:
        tf.summary.scalar("loss", self.loss)

        return

    # ------------------------------------------------------------------------------------------------------------------
    def define_initializer(self, args):

        if args.initialization_mode == 'load-pretrained':

            # self.model_variables has all the model variables (it doesn't include the optimizer variables
            # or others).
            # We will filter out from it the variables that fit in the scopes specified in args.modified_scopes.
            # Usually this is necessary in the last layer, if the number of outputs is different.
            assert type(args.modified_scopes) == list, 'modified_scopes should be a list.'
            varnames_to_restore = []

            print('')
            logging.debug('Variables to restore:')
            candidate_vars_to_restore = self.model_variables
            if args.restore_optimizer:
                candidate_vars_to_restore.extend([var.name for var in self.optimizer_variables])
            for var in candidate_vars_to_restore:
                is_modified = False
                for modscope in args.modified_scopes:
                    if modscope in var:
                        is_modified = True
                if not is_modified:
                    varnames_to_restore.append(var)
                    logging.debug(var)

            # Get the variables to restore:
            vars_to_restore = tf.contrib.framework.get_variables_to_restore(include=varnames_to_restore)
            self.restore_fn = tf.contrib.framework.assign_from_checkpoint_fn(args.weights_file, vars_to_restore)

            # Variables to initialize from scratch (the rest):
            vars_new = tf.contrib.framework.get_variables_to_restore(exclude=varnames_to_restore)
            self.init_op = tf.variables_initializer(vars_new)

        elif args.initialization_mode == 'scratch':
            self.init_op = tf.global_variables_initializer()

        else:
            raise Exception('Initialization mode not recognized.')

    # ------------------------------------------------------------------------------------------------------------------
    def initialize(self, sess, args):

        if args.initialization_mode == 'load-pretrained':
            self.restore_fn(sess)
            sess.run(self.init_op)
        elif args.initialization_mode == 'scratch':
            sess.run(self.init_op)
        else:
            raise Exception('Initialization mode not recognized.')

    # ------------------------------------------------------------------------------------------------------------------
    def build_optimizer(self, args):

        # Choose the variables to train:
        vars_to_train = tools.get_trainable_variables(args)

        print('')
        logging.debug('Training variables:')

        for v in vars_to_train:
            logging.debug(v.name)

        # if self.nbatches_accum > 0:
        #     args.learning_rate = args.learning_rate / self.nbatches_accum

        self.learning_rate = tf.Variable(initial_value=args.learning_rate, dtype=tf.float32, name='learning_rate')
        self.lr_to_update = tf.placeholder(dtype=tf.float32, shape=())
        self.update_lr_op = tf.assign(self.learning_rate, self.lr_to_update, name='UpdateLR')

        previous_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # Decide what optimizer to use:
        if args.optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif args.optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif args.optimizer_name == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif args.optimizer_name == 'momentum':
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, args.momentum)
        else:
            raise Exception('Optimizer name not recognized.')

        update_bn_stats_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        if self.nbatches_accum > 0:
            # assert args.optimizer_name == 'sgd', 'If nbatches_accum > 0, the optimizer must be SGD'
            accum_grads = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in vars_to_train]
            self.zero_ops = [ag.assign(tf.zeros_like(ag)) for ag in accum_grads]
            gvs = optimizer.compute_gradients(self.loss, vars_to_train)
            self.train_step = optimizer.apply_gradients([(accum_grads[i], gv[1]) for i, gv in enumerate(gvs)])
            with tf.control_dependencies(update_bn_stats_ops):
                self.accum_ops = [accum_grads[i].assign_add(gv[0] / float(self.nbatches_accum)) for i, gv in enumerate(gvs)]

        else:
            with tf.control_dependencies(update_bn_stats_ops):
                self.train_op = optimizer.minimize(self.loss, var_list=vars_to_train, name='train_op')

        posterior_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.optimizer_variables = []
        for var in posterior_variables:
            if var not in previous_variables:
                self.optimizer_variables.append(var)

        return


def prepare_tensorboard(sess, outdir):
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(os.path.join(outdir, 'tensorboard'), sess.graph)
    python_dir = os.path.dirname(sys.executable)
    tensorboard_path = os.path.join(python_dir, 'tensorboard')
    command = tensorboard_path + ' --logdir=' + os.path.join(outdir, 'tensorboard')
    if os.name == 'nt':  # Windows
        subprocess.Popen(["start", "cmd", "/k", command], shell=True)
    elif os.name == 'posix':  # Ubuntu
        os.system('gnome-terminal -e "bash -c \'' + command + '\'; $SHELL"')
    else:
        raise Exception('Operative system name not recognized: ' + str(os.name))
    hostname = socket.gethostname()
    tensorboard_url = 'http://' + hostname + ':6006'
    return merged, summary_writer, tensorboard_url
