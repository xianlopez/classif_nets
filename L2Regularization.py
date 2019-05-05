import tensorflow as tf
import tools
import logging

def L2RegularizationLoss(args):
    print('')
    logging.info('L2 regularization loss.')
    vars_to_train = tools.get_trainable_variables(args)
    reg_loss = tf.zeros((), dtype=vars_to_train[0].dtype)
    for var in vars_to_train:
        add_to_reg = True
        for string_not_add in args.vars_to_skip_l2_reg:
            if string_not_add in var.name:
                add_to_reg = False
                break
        if add_to_reg:
            logging.info('Adding ' + var.name + ' to regularization loss')
            reg_loss += tf.reduce_sum(tf.square(var))
        else:
            logging.info('Not adding ' + var.name + ' to regularization loss')
    # reg_loss *= args.l2_regularization / 2.0
    reg_loss *= args.l2_regularization
    # reg_loss = tf.Print(reg_loss, [reg_loss], 'reg loss')
    tf.summary.scalar('reg_loss', reg_loss)
    return reg_loss