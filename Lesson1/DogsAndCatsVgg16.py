from __future__ import print_function, division
import os, json
from glob import glob
import numpy as np
import pandas
from keras.preprocessing.image import load_img, img_to_array

np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
from utils import plots
import vgg16

reload(vgg16)
from vgg16 import Vgg16

path_train = "../Data/Lesson1/dogscats/"
path_pre_test = "../Data/Lesson1/dogscats/test1/"

# As large as you can, but no larger than 64 is recommended.
# If you have an older or cheaper GPU, you'll run out of memory, so will have to decrease this.
batch_size = 16

vgg = Vgg16()
# Grab a few images at a time for training and validation.
# NB: They must be in subdirectories named based on their category
batches = vgg.get_batches(path_train + 'train', batch_size=batch_size)
val_batches = vgg.get_batches(path_train + 'valid', batch_size=batch_size * 2, shuffle=False)

vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=1)
pred_result_list = []
file_list = os.listdir(path_pre_test)

for file_name in file_list:
    probs = vgg.model.predict(
        img_to_array(load_img(path_pre_test + file_name, target_size=[224, 224])).reshape(1, 3, 224, 224))

    number = int(file_name.split('.')[0])
    pred_result_list.append([number, np.argmax(probs, axis=1)[0]])


pred_result = pandas.DataFrame(pred_result_list, columns=['id', 'label'])
pred_result.sort_values(by=['id'], ascending=True,inplace=True)

pred_result.to_csv('result.csv', index=False)
