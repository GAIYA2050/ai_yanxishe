# coding=utf-8
# author=yphacker

import os

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data")

train_path = os.path.join(data_path, "train.csv")
wav_train_path = os.path.join(data_path, "train")
wav_test_path = os.path.join(data_path, "test")

model_path = os.path.join(data_path, "model")
model_save_path = os.path.join(model_path, "cnn.bin")

num_classes = 50
batch_size = 64
epochs_num = 50
train_print_step = 1
patience_epoch = 8
