from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential
from keras.utils.data_utils import get_file
import json
import numpy as np
from keras.preprocessing import image
from utils import plots


def ConvBlock(layers, model, filters):
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(filters, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


def FCBlock(model):
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))


# Mean of each channel as provided by VGG researchers
vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((3, 1, 1))


def vgg_preprocess(x):
    x = x - vgg_mean  # subtract mean
    return x[:, ::-1]  # reverse axis bgr->rgb


def VGG_16():
    model = Sequential()
    model.add(Lambda(vgg_preprocess, input_shape=(3, 224, 224)))

    ConvBlock(2, model, 64)
    ConvBlock(2, model, 128)
    ConvBlock(3, model, 256)
    ConvBlock(3, model, 512)
    ConvBlock(3, model, 512)

    model.add(Flatten())
    FCBlock(model)
    FCBlock(model)
    model.add(Dense(1000, activation='softmax'))
    return model


FILES_PATH = 'http://files.fast.ai/models/'
CLASS_FILE = 'imagenet_class_index.json'
# Keras' get_file() is a handy function that downloads files, and caches them for re-use later
fpath = get_file(CLASS_FILE, FILES_PATH + CLASS_FILE, cache_subdir='models')
with open(fpath) as f:
    class_dict = json.load(f)
# Convert dictionary with string indexes into an array
classes = [class_dict[str(i)][1] for i in range(len(class_dict))]
print classes[:5]

model = VGG_16()
fpath = get_file('vgg16.h5', FILES_PATH + 'vgg16.h5', cache_subdir='models')
model.load_weights(fpath)
batch_size = 4
path = "../Data/Lesson1/dogscats/sample/"


def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True,
                batch_size=batch_size, class_mode='categorical'):
    return gen.flow_from_directory(path + dirname, target_size=(224, 224),
                                   class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


batches = get_batches('train', batch_size=batch_size)
val_batches = get_batches('valid', batch_size=batch_size)
imgs, labels = next(batches)

# This shows the 'ground truth'
plots(imgs, titles=labels)

def pred_batch(imgs):
    preds = model.predict(imgs)
    idxs = np.argmax(preds, axis=1)
    print('Shape: {}'.format(preds.shape))
    print('First 5 classes: {}'.format(classes[:5]))
    print('First 5 probabilities: {}\n'.format(preds[0, :5]))
    print('Predictions prob/class: ')

    for i in range(len(idxs)):
        idx = idxs[i]
        print ('  {:.4f}/{}'.format(preds[i, idx], classes[idx]))


pred_batch(imgs)
