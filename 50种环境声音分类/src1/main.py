# # coding=utf-8
# # author=yphacker
#
# import os
# import time
# import argparse
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import torch
# from torch import nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from conf import config
# from model.cnn import Model
# from utils.data_utils import MyDataset
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#
# def evaluate(model, val_loader, criterion):
#     model.eval()
#     data_len = 0
#     total_loss = 0
#     total_acc = 0
#     with torch.no_grad():
#         for batch_x, batch_y in val_loader:
#             batch_len = len(batch_y)
#             # batch_len = len(batch_y.size(0))
#             data_len += batch_len
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#             probs = model(batch_x)
#             loss = criterion(probs, batch_y)
#             total_loss += loss.item()
#             _, preds = torch.max(probs, 1)
#             total_acc += (preds == batch_y).sum().item()
#
#     return total_loss / data_len, total_acc / data_len
#
#
# def train(train_data, val_data):
#     train_dataset = MyDataset(train_data, 'train')
#     train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
#     val_dataset = MyDataset(val_data, 'val')
#     val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False)
#
#     model = Model().to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#
#     best_val_acc = 0
#     last_improved_epoch = 0
#     for cur_epoch in range(config.epochs_num):
#         start_time = int(time.time())
#         model.train()
#         print('epoch:{}, step:{}'.format(cur_epoch + 1, len(train_loader)))
#         cur_step = 0
#         for batch_x, batch_y in train_loader:
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#
#             optimizer.zero_grad()
#             probs = model(batch_x)
#
#             train_loss = criterion(probs, batch_y)
#             train_loss.backward()
#             optimizer.step()
#
#             cur_step += 1
#             if cur_step % config.train_print_step == 0:
#                 _, train_preds = torch.max(probs, 1)
#                 train_corrects = (train_preds == batch_y).sum().item()
#                 train_acc = train_corrects * 1.0 / len(batch_y)
#                 msg = 'the current step: {0}/{1}, train loss: {2:>5.2}, train acc: {3:>6.2%}'
#                 print(msg.format(cur_step, len(train_loader), train_loss.item(), train_acc))
#         val_loss, val_acc = evaluate(model, val_loader, criterion)
#         if val_acc >= best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), config.model_save_path)
#             improved_str = '*'
#             last_improved_epoch = cur_epoch
#         else:
#             improved_str = ''
#         # msg = 'the current epoch: {0}/{1}, train loss: {2:>5.2}, train acc: {3:>6.2%},  ' \
#         #       'val loss: {4:>5.2}, val acc: {5:>6.2%}, {6}'
#         msg = 'the current epoch: {0}/{1}, val loss: {2:>5.2}, val acc: {3:>6.2%}, cost: {4}s {5}'
#         end_time = int(time.time())
#         print(msg.format(cur_epoch + 1, config.epochs_num, val_loss, val_acc,
#                          end_time - start_time, improved_str))
#         # scheduler.step()
#         if cur_epoch - last_improved_epoch > config.patience_epoch:
#             print("No optimization for a long time, auto-stopping...")
#             break
#
#
# def predict():
#     model = Model().to(device)
#     model.load_state_dict(torch.load(config.model_save_path))
#     data_len = len(os.listdir(config.wav_test_path))
#     test_path_list = ['{}/{}.wav'.format(config.wav_test_path, x) for x in range(0, data_len)]
#     test_data = pd.DataFrame({'filename': test_path_list})
#     test_dataset = MyDataset(test_data, 'test')
#     test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
#     model.eval()
#     preds_list = []
#     with torch.no_grad():
#         for batch_x, _ in test_loader:
#             batch_x = batch_x.to(device)
#             probs = model(batch_x)
#             # pred = torch.argmax(output, dim=1)
#             _, preds = torch.max(probs, 1)
#             preds_list += [p.item() for p in preds]
#
#     submission = pd.DataFrame({"id": range(len(preds_list)), "label": preds_list})
#     submission.to_csv('../data/submission.csv', index=False, header=False)
#
#
# def main(op):
#     if op == 'train':
#         train_df = pd.read_csv('../data/train.csv')
#         train_df = train_df[:10]
#         # print(train_df['label'].value_counts())
#         train_data, val_data = train_test_split(train_df, shuffle=True, test_size=0.1)
#         print('train:{}, val:{}'.format(train_data.shape[0], val_data.shape[0]))
#         train(train_data, val_data)
#     elif op == 'eval':
#         pass
#     elif op == 'predict':
#         predict()
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
#     parser.add_argument("-b", "--BATCH", default=1024, type=int, help="batch size")
#     parser.add_argument("-e", "--EPOCHS", default=50, type=int, help="train epochs")
#     # parser.add_argument("-m", "--MODEL", default='cnn', type=str, help="model select")
#     args = parser.parse_args()
#     config.batch_size = args.BATCH
#     config.epochs_num = args.EPOCHS
#
#     main(args.operation)

import os
from fastai.vision import *
from fastai.vision.models.wrn import wrn_22
from fastai.vision.models import *

from torchvision.models import *
# import pretrainedmodels
from efficientnet_pytorch import EfficientNet

# from nb_new_data_augmentation import *
SEED = 2019


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)
from conf import config


def get_data(sz, bs):
    tfms = get_transforms(flip_vert=False, max_rotate=10, p_affine=1., p_lighting=1.,
                          max_lighting=0.7, max_zoom=1.5, max_warp=0.1)
    # tfms = get_transforms(flip_vert=False, max_rotate=10, p_affine =1., p_lighting=1.,
    # max_lighting=0.2, max_zoom=1.2, max_warp=0.1)
    # tfms = get_transforms(flip_vert=False,do_flip=True, max_rotate=30, p_affine =0.5,
    # max_zoom=1.2, max_warp=0.1)
    # tfms = get_transforms()
    data = ImageDataBunch.from_folder(config.data_path, size=sz, train='image_train', valid='image_train',
                                      test='image_test', bs=bs)
    data.normalize(imagenet_stats)
    return data


data = get_data(448, 16)
# data.show_batch(row =6,figsize = (15,15))

arch = EfficientNet.from_pretrained('efficientnet-b4', num_classes=data.c)
# arch = EfficientNet.from_pretrained('efficientnet-b3',num_classes=data.c)
# arch = EfficientNet.from_pretrained('efficientnet-b7',num_classes=data.c)
# loss_func = LabelSmoothingCrossEntropy()
learn = Learner(data, arch, metrics=accuracy)
# learn = learn.split([learn.model._conv_stem,loss_func = loss_func,learn.model._blocks,learn.model._conv_head])
learn = learn.split([learn.model._conv_stem, learn.model._blocks, learn.model._conv_head]).mixup()
learn.to_fp16()
learn.lr_find()
learn.recorder.plot()

classes = learn.data.classes
classes

lr = 1e-2
learn.fit_one_cycle(15, (1e-4, 1e-3, 1e-2), wd=0.2)
# learn.fit_one_cycle(16,lr, wd=0.2)
learn.recorder.plot_losses()
learn.recorder.plot_lr()

learn.unfreeze()
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(12, max_lr=(1e-5, 1e-4, 1e-3), wd=(1e-4, 1e-3, 0.2))
learn.recorder.plot_losses()
learn.recorder.plot_lr()

# test = ImageList.from_folder('{}/test_nf'.format(config.data_path))
# len(test)

learn.to_fp32()
# preds, _ = learn.TTA(ds_type=DatasetType.Test)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)
classes = learn.data.classes

results = ['' for x in range(preds.shape[0])]
for i in np.arange(preds.shape[0]):
    # results[i] = preds[i].numpy()
    # results[i] = str(results[i][0]) + ',' + str(results[i][1])
    results[i] = classes[np.argmax(preds[i]).numpy()]
ids = [item.name[:-4] for item in learn.data.test_ds.items]
df = pd.DataFrame({'id': ids, 'label': results})
df.to_csv('submission.csv', index=False, header=False)
# df = pd.DataFrame(results, ids,columns=['id','label'])
print(df.head())
