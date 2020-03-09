# coding=utf-8
# author=yphacker

import os

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data")
model_path = os.path.join(data_path, "model")
submission_path = os.path.join(data_path, "submission")
for path in [model_path, submission_path]:
    if not os.path.isdir(path):
        os.makedirs(path)

image_train_path = os.path.join(data_path, 'train')
image_test_path = os.path.join(data_path, 'test')
train_path = os.path.join(data_path, 'train.csv')
test_path = os.path.join(data_path, 'test.csv')

num_classes = 4
# img_size = 224
img_size = 400
batch_size = 32
epochs_num = 8
train_print_step = 50
patience_epoch = 4
