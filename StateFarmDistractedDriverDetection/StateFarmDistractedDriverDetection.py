from keras.preprocessing.image import img_to_array, load_img
import numpy as np

import vgg16

reload(vgg16)
from vgg16 import Vgg16
import os
import pandas

path_train = '../Data/StateFarmDistractedDriverDetection/imgs/'
path_pre_test = '../Data/StateFarmDistractedDriverDetection/imgs/sample1_100/'
batch_size = 8
vgg = Vgg16()
batches = vgg.get_batches(path_train + 'train', batch_size=batch_size)
valid_batches = vgg.get_batches(path_train + 'valid', batch_size=batch_size)
vgg.finetune(batches)
vgg.fit(batches, valid_batches, nb_epoch=1)

'''
pred_result_list = []
file_list = os.listdir(path_pre_test)

for file_name in file_list:
    probs = vgg.model.predict(
        img_to_array(load_img(path_pre_test + file_name, target_size=[224, 224])).reshape(1, 3, 224, 224))

    number = int(file_name.split('.')[0])
    pred_result_list.append([number, np.argmax(probs, axis=1)[0]])

pred_result = pandas.DataFrame(pred_result_list, columns=['id', 'label'])
pred_result.sort_values(by=['id'], ascending=True, inplace=True)

pred_result.to_csv('result.csv', index=False)
'''