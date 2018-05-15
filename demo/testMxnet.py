import mxnet as mx
import numpy as np
import cv2
from collections import namedtuple


Batch = namedtuple('Batch', ['data'])

# Load the symbols for the networks
with open('classes.txt', 'r') as f:
    synsets = [l.rstrip() for l in f]

# Load the network parameters
sym, arg_params, aux_params = mx.model.load_checkpoint('read-to-me', 0)

# Load the network into an MXNet module and bind the corresponding parameters
mod = mx.mod.Module(symbol=sym, context=mx.gpu(0))
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 300, 300))])
mod.set_params(arg_params, aux_params)


def filter_positive_detections(detections):
    """
    First column (class id) is -1 for negative detections
    :param detections:
    :return:
    """
    class_idx = 0
    assert (isinstance(detections, mx.nd.NDArray) or isinstance(detections, np.ndarray))
    detections_per_image = []
    # for each image
    for i in range(detections.shape[0]):
        result = []
        det = detections[i, :, :]
        for obj in det:
            if obj[class_idx] >= 0:
                result.append(obj)
        detections_per_image.append(result)
    logging.info("%d positive detections", len(result))
    return detections_per_image


'''
Function to predict objects by giving the model a pointer to an image file and running a forward pass through the model.

inputs:
filename = jpeg file of image to classify objects in
mod = the module object representing the loaded model
synsets = the list of symbols representing the model
N = Optional parameter denoting how many predictions to return (default is top 5)

outputs:
python list of top N predicted objects and corresponding probabilities
'''


def predict(image, reshape=(300, 300)):
    topN = []

    # Switch RGB to BGR format (which ImageNet networks take)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if img is None:
        return topN

    # Resize image to fit network input
    img = cv2.resize(img, reshape)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]

    # Run forward on the image
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    return prob[a[0]]
