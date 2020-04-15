# coding=utf-8
# author=yphacker


import os
import pandas as pd
from conf import config

# create spectrogram function and test image

import os
import imageio

import matplotlib

matplotlib.use('agg')

from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import pylab

import librosa
from librosa import display
import numpy as np


# src = '/data/zhangjianming/kaggle/sound/data/train_wav50/0/1-30226-A-0.wav'
# dest = "/data/zhangjianming/kaggle/sound/data/test.jpg"
#
#
def create_spectrogram(source_filepath, destination_filepath):
    y, sr = librosa.load(source_filepath, sr=22050)  # Use the default sampling rate of 22,050 Hz

    # Pre-emphasis filter
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # Compute spectrogram
    M = librosa.feature.melspectrogram(y,
                                       sr,
                                       fmax=sr / 2,  # Maximum frequency to be used on the on the MEL scale
                                       n_fft=2048,
                                       hop_length=512,
                                       n_mels=96,  # As per the Google Large-scale audio CNN paper
                                       power=2)  # Power = 2 refers to squared amplitude
    # Power in DB
    log_power = librosa.power_to_db(M, ref=np.max)  # Covert to dB (log) scale

    # Plotting the spectrogram and save as JPG without axes (just the image)
    pylab.figure(figsize=(5, 5))  # was 14, 5
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
    librosa.display.specshow(log_power, cmap=cm.jet)
    print(destination_filepath)
    pylab.savefig(destination_filepath, bbox_inches=None, pad_inches=0)
    pylab.close()


#
#
# create_spectrogram(src, dest)

import os


def get_train():
    root_dir = "../data/train_nf/"
    for root, dirs, filenames in os.walk('../data/train/'):
        for filename in filenames:
            src = root + '/' + filename
            label = filename.split('-')[-1].split('.')[0]
            # print(src,label)
            dest = root_dir + label + '/' + filename.split('.')[0] + '.jpg'
            path = root_dir + label
            if not os.path.isdir(path):
                os.makedirs(path)
            # print(dest)
            create_spectrogram(src, dest)

    print("done!")


def get_test():
    root_dir = "../data/test_nf/"
    for root, dirs, filenames in os.walk('../data/test/'):
        for filename in filenames:
            src = root + filename
            print(src)
            dest = root_dir + filename.split('.')[0] + '.jpg'
            path = root_dir
            if not os.path.isdir(path):
                os.makedirs(path)
            print(dest)
            create_spectrogram(src, dest)


if __name__ == '__main__':
    # get_train()
    get_test()
