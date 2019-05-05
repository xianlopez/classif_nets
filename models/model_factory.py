# ======================================================================================================================
from models import vgg16
from models import vgg16_f16
from models import vggA
from models import resnet50
from models import alexnet
from models import inception_resnet_v1
from models import inception_v3
from models import nasnet
import tensorflow as tf


# ----------------------------------------------------------------------------------------------------------------------
def build_model(inputs, nclasses, args):

    # When building a model, we need three things:
    # net_output: the output of the network. This is what will feed the loss functions.
    # predictions: it may be the same as net_output, but not always. Sometimes, what is used by the loss is not easily
    #           understandable to get the predictions that are really desired. In this case, predictions is a post-
    #           process of net_output, that has a shape easier to understand.
    is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')
    if args.model_name == 'vgg16':
        net_output, predictions = vgg16.build(inputs, nclasses, is_training)
        predictions = tf.nn.softmax(predictions)

    elif args.model_name == 'vgg16_f16':
        net_output, predictions = vgg16_f16.build(inputs, nclasses, is_training)
        predictions = tf.nn.softmax(predictions)

    elif args.model_name == 'vggA':
        net_output, predictions = vggA.build(inputs, nclasses, is_training)
        predictions = tf.nn.softmax(predictions)

    elif args.model_name == 'nasnet_large':
        net_output, predictions = nasnet.build(inputs, nclasses, is_training, True)
        predictions = tf.nn.softmax(predictions)

    elif args.model_name == 'nasnet_mobile':
        net_output, predictions = nasnet.build(inputs, nclasses, is_training, False)
        predictions = tf.nn.softmax(predictions)

    elif args.model_name == 'resnet50':
        net_output, predictions = resnet50.build(inputs, nclasses, is_training)
        predictions = tf.nn.softmax(predictions)

    elif args.model_name == 'alexnet':
        net_output, predictions = alexnet.build(inputs, nclasses)

    elif args.model_name == 'inception_resnet_v1':
        net_output, predictions = inception_resnet_v1.build(inputs, nclasses, is_training)

    elif args.model_name == 'inception_v3':
        net_output, predictions = inception_v3.build(inputs, nclasses, is_training)

    else:
        raise Exception('Model name not recognized.')

    # model_variables is a list with all the variables that the model defines. This is used later to decide what layers
    # to restore from pre-trained weights, or which ones to train or not.
    model_variables = [n.name for n in tf.global_variables()]

    return net_output, predictions, model_variables, is_training


# ----------------------------------------------------------------------------------------------------------------------
def define_input_shape(args):

    if args.model_name == 'vgg16':
        input_width = 224
        input_height = 224
    elif args.model_name == 'vgg16_f16':
        input_width = 224
        input_height = 224
    elif args.model_name == 'vggA':
        input_width = 224
        input_height = 224
    elif args.model_name == 'nasnet_large':
        input_width = 331
        input_height = 331
    elif args.model_name == 'nasnet_mobile':
        input_width = 224
        input_height = 224
    elif args.model_name == 'resnet50':
        input_width = 224
        input_height = 224
    elif args.model_name == 'alexnet':
        input_width = 224
        input_height = 224
    elif args.model_name == 'inception_resnet_v1':
        input_width = 160
        input_height = 160
    elif args.model_name == 'inception_v3':
        input_width = 299
        input_height = 299
    else:
        raise Exception('Model name not recognized.')

    input_shape = [input_width, input_height]

    return input_shape

