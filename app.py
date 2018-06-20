import sys
from os.path import isfile

import numpy as np
import cv2

NET_ASSETS_DIR = "./net-assets"


def get_image_from_file(image_path):
    if None is image_path or not isfile(image_path):
        raise Exception("'image_path' has wrong value!")
    return cv2.imread(image_path)


# def get_image_from_base64(src_str):
#     import base64
#     r = base64.decodestring(src_str)
#     jpg_as_np = np.frombuffer(r, dtype=np.uint8)
#     return jpg_as_np

def get_image_from_base64(base64_str):
    import re
    import base64
    from io import BytesIO
    from PIL import Image

    if not isinstance(base64_str, basestring):
        return None
    base64_str = base64_str.replace('\n', '')
    base64_str = re.sub("data:image/[\w]+;base64,", "", base64_str)

    im = Image.open(BytesIO(base64.b64decode(base64_str)))

    return np.array(im)

    # b64decode = base64.b64decode(base64_str)
    # b = BytesIO(b64decode)
    # jpg_as_np = Image.frombuffer(b64decode)
    # return jpg_as_np

    # Image.frombytes()

    # b = BytesIO(b64decode)
    # return Image.open(b)


def get_image_classificators(image, limit=5):
    # read names of classes
    with open(NET_ASSETS_DIR + '/synset_words.txt') as f:
        classes = [x[x.find(' ') + 1:] for x in f]

    # create tensor with 224x224 spatial size and subtract mean values (104, 117, 123)
    # from corresponding channels (R, G, B)
    blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

    # load model from caffe
    net = cv2.dnn.readNetFromCaffe(
        NET_ASSETS_DIR + '/bvlc_googlenet.prototxt',
        NET_ASSETS_DIR + '/bvlc_googlenet.caffemodel'
    )
    # feed input tensor to the model
    net.setInput(blob)
    # perform inference and get output
    out = net.forward()
    # get indices with the highest probability
    indexes = np.argsort(out[0])[-limit:]

    result = []
    for i in reversed(indexes):
        result.append({'class': classes[i].strip(), 'probability': float(out[0][i]), })

    return result


if __name__ == '__main__':
    filename, = sys.argv[1:]
    print get_image_classificators(get_image_from_file(filename), 10)
