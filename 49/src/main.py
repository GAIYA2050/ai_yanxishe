# coding=utf-8
# author=yphacker

import os
import time
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from conf import config
from conf import model_config_bert as model_config
from model.bert import Model
from utils.bert_data_utils import MyDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(model, val_loader, criterion):
    model.eval()
    data_len = 0
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_len = len(batch_y)
            # batch_len = len(batch_y.size(0))
            data_len += batch_len
            # batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            input_ids, token_type_ids = batch_x
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            batch_y = batch_y.to(device)
            probs = model(input_ids, token_type_ids=token_type_ids)
            loss = criterion(probs, batch_y)
            total_loss += loss.item()
            _, preds = torch.max(probs, 1)
            total_acc += (preds == batch_y).sum().item()

    return total_loss / data_len, total_acc / data_len


def train(train_data, val_data):
    train_dataset = MyDataset(train_data, 'train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataset = MyDataset(val_data, 'val')
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False)

    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    best_val_acc = 0
    for cur_epoch in range(config.epochs_num):
        start_time = int(time.time())
        model.train()
        print('epoch:{}, step:{}'.format(cur_epoch + 1, len(train_loader)))
        cur_step = 0
        for batch_x, batch_y in train_loader:
            input_ids, token_type_ids = batch_x
            # batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            probs = model(input_ids, token_type_ids=token_type_ids)

            train_loss = criterion(probs, batch_y)
            train_loss.backward()
            optimizer.step()

            cur_step += 1
            if cur_step % config.train_print_step == 0:
                _, train_preds = torch.max(probs, 1)
                train_corrects = (train_preds == batch_y).sum().item()
                train_acc = train_corrects * 1.0 / len(batch_y)
                msg = 'the current step: {0}/{1}, train loss: {2:>5.2}, train acc: {3:>6.2%}'
                print(msg.format(cur_step, len(train_loader), train_loss.item(), train_acc))
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.model_save_path)
            improved_str = '*'
        else:
            improved_str = ''
        msg = 'the current epoch: {0}/{1}, val loss: {2:>5.2}, val acc: {3:>6.2%}, cost: {4}s {5}'
        end_time = int(time.time())
        print(msg.format(cur_epoch + 1, config.epochs_num, val_loss, val_acc,
                         end_time - start_time, improved_str))
        scheduler.step()


def predict():
    model = Model().to(device)
    model.load_state_dict(torch.load(config.model_save_path))
    test_data = pd.read_csv(config.test_path)
    test_dataset = MyDataset(test_data, 'test')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    model.eval()
    preds_list = []
    with torch.no_grad():
        for batch_x, _ in tqdm(test_loader):
            # batch_x = batch_x.to(device)
            input_ids, token_type_ids = batch_x
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            probs = model(input_ids, token_type_ids=token_type_ids)
            # pred = torch.argmax(output, dim=1)
            _, preds = torch.max(probs, 1)
            preds_list += [p.cpu().data.numpy() for p in preds]

    submission = pd.DataFrame({"id": range(len(preds_list)), "label": preds_list})
    submission.to_csv('../data/submission.csv', index=False, header=False)


def main(op):
    if op == 'train':
        train_df = pd.read_csv(config.train_path)
        print(train_df['label'].value_counts())
        train_data, val_data = train_test_split(train_df, shuffle=True, test_size=0.1)
        print('train:{}, val:{}'.format(train_data.shape[0], val_data.shape[0]))
        train(train_data, val_data)
    elif op == 'eval':
        pass
    elif op == 'predict':
        predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
    parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("-e", "--epochs_num", default=2, type=int, help="epochs num")
    args = parser.parse_args()
    config.batch_size = args.batch_size
    config.epochs_num = args.epochs_num

    main(args.operation)
