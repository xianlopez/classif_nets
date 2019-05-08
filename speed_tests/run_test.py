import argparse
import tensorflow as tf
from speed_tests.speed_test_1 import speed_test_1
from speed_tests.speed_test_2 import speed_test_2
from speed_tests.speed_test_3 import speed_test_3
from speed_tests.speed_test_4 import speed_test_4
from speed_tests.speed_test_5 import speed_test_5
from speed_tests.speed_test_6 import speed_test_6
from speed_tests.speed_test_7 import speed_test_7


def parse_args():
    parser = argparse.ArgumentParser(description='Speed tests.')
    parser.add_argument('-t', '--test', type=int, default=None,
                        help='Test number')
    parser.add_argument('-d', '--device', type=str, default='gpu0',
                        help='Device on which to execute the test (gpu0, gpu1, cpu)')
    arguments = parser.parse_args()

    assert arguments.test is not None, 'Please specify a test number'
    return arguments


def run_test(test_num):
    if test_num == 1:
        speed_test_1()
    elif test_num == 2:
        speed_test_2()
    elif test_num == 3:
        speed_test_3()
    elif test_num == 4:
        speed_test_4()
    elif test_num == 5:
        speed_test_5()
    elif test_num == 6:
        speed_test_6()
    elif test_num == 7:
        speed_test_7()
    else:
        raise Exception('Test number not recognized.')


def main(args):
    if args.device == 'gpu0':
        run_test(args.test)
    elif args.device == 'gpu1':
        with tf.device('/gpu:1'):
            run_test(args.test)
    elif args.device == 'cpu':
        with tf.device('/cpu:0'):
            run_test(args.test)
    else:
        raise Exception('Device not recognized.')


if __name__ == "__main__":
    args = parse_args()
    main(args)

