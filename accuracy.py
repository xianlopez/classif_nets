import numpy as np


def compute_accuracy(predictions, labels):
    nimages = len(predictions)
    if nimages > 0:
        ncorrect = 0
        for i in range(nimages):
            predicted_class = np.argmax(np.array(predictions[i]))
            if predicted_class == labels[i]:
                ncorrect += 1
        accuracy = ncorrect / nimages
    else:
        accuracy = 0
    return accuracy
