# coding=utf-8
# author=yphacker

import argparse
import keras
import pandas as pd
from conf import config
from mrcnn.model import MaskRCNN
from load_data import MyDataset


def train():
    df = pd.read_csv(config.train_path)
    df = df.sample(frac=1).reset_index(drop=True)
    x_train = df['img_path'].values.tolist()
    y_train = df['box'].values.tolist()
    dev_sample_index = -1 * int(0.1 * float(len(y_train)))
    x_train, x_val = x_train[:dev_sample_index], x_train[dev_sample_index:]
    y_train, y_val = y_train[:dev_sample_index], y_train[dev_sample_index:]
    print('train:{}, val:{}, all:{}'.format(len(y_train), len(y_val), df.shape[0]))

    # load the train dataset
    train_set = MyDataset()
    train_set.load_dataset(x_train, y_train, mode='train')
    train_set.prepare()

    # load the val dataset
    val_set = MyDataset()
    val_set.load_dataset(x_val, y_val, mode='val')
    val_set.prepare()

    model_config = config.TrainConfig()
    # define the model
    model = MaskRCNN(mode='training', model_dir='./', config=model_config)
    model.keras_model.metrics_tensors = []

    # load weights (mscoco) and exclude the output layers
    model.load_weights(config.mrcnn_model_path, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    # save checkpoint
    ModelCheckpoint = keras.callbacks.ModelCheckpoint(config.keras_model_dir, verbose=0, save_best_only=True,
                                                      save_weights_only=True)
    callbacks = [ModelCheckpoint]
    model.train(train_set, val_set, learning_rate=model_config.LEARNING_RATE, epochs=config.epochs_num, layers='heads',
                custom_callbacks=callbacks)


def predict():
    import numpy as np
    from mrcnn.model import MaskRCNN
    from mrcnn.model import mold_image
    import skimage.io

    model_config = config.PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir='./', config=model_config)
    model.keras_model.metrics_tensors = []
    # load model weights
    model.load_weights(config.keras_model_dir, by_name=True)

    dir_path = '../data/test'
    outfile = open('../data/submission.csv', 'w')
    for id in range(0, 1592):
        try:
            file_path = '{}/{}.jpg'.format(dir_path, id)
            image = skimage.io.imread(file_path)
            scaled_image = mold_image(image, model_config)
            sample = np.expand_dims(scaled_image, 0)
            pred = model.detect(sample, verbose=0)[0]
        except:
            print(file_path)
            # 报错的话，先赋值1，0
            outfile.write('{},{},{}\n'.format(id, 1, 0))
            continue
        # 按照得分进行排序
        indices = np.argsort(pred["scores"])[::-1]
        boxes = []
        for i in range(len(indices)):
            boxes.append([pred["class_ids"][i] - 1, pred['rois'][i][1], pred['rois'][i][0], pred['rois'][i][3],
                          pred['rois'][i][2]])
        boxes = np.array(boxes)
        boxes = boxes[indices]
        face = 0
        face_mask = 0
        for box in boxes:
            label = box[0]
            if label == 0:
                face += 1
            else:
                face_mask += 1
        outfile.write('{},{},{}\n'.format(id, face, face_mask))
    outfile.close()


def main(op):
    if op == 'train':
        train()
    elif op == 'predict':
        predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
    parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
    parser.add_argument("-e", "--EPOCHS", default=8, type=int, help="train epochs")
    args = parser.parse_args()
    config.batch_size = args.BATCH
    config.epochs_num = args.EPOCHS
    main(args.operation)
