# coding=utf-8
# author=yphacker

import pandas as pd
from xml.dom.minidom import parse
import xml.dom.minidom
from conf import config


def solve():
    img_paths = []
    boxes = []
    dir_path = '../data/train'
    for i in range(6367):
        file_path = '{}/{}.xml'.format(dir_path, i)
        dom = xml.dom.minidom.parse(file_path)
        root = dom.documentElement
        objs = root.getElementsByTagName('object')
        box = []
        for obj in objs:
            xmin = obj.getElementsByTagName('xmin')[0].childNodes[0].data
            ymin = obj.getElementsByTagName('ymin')[0].childNodes[0].data
            xmax = obj.getElementsByTagName('xmax')[0].childNodes[0].data
            ymax = obj.getElementsByTagName('ymax')[0].childNodes[0].data
            name = obj.getElementsByTagName('name')[0].childNodes[0].data
            # print(xmin, ymin, xmax, ymax, name)
            box.append('{},{},{},{},{}'.
                       format(xmin, ymin, xmax, ymax, 0 if name == 'face' or name == 'face_mask' else 1))

        img_path = '{}/{}.jpg'.format(config.image_train_path, i)
        img_paths.append(img_path)
        boxes.append(' '.join(box))
    df = pd.DataFrame({'img_path': img_paths, 'box': boxes})
    df.to_csv(config.train_path, index=None)


if __name__ == '__main__':
    solve()
