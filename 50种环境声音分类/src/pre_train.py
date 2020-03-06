# coding=utf-8
# author=yphacker


import os
import pandas as pd
from conf import config

if __name__ == '__main__':
    filenames = [filename for filename in os.listdir(config.wav_train_path)]
    train_df = pd.DataFrame({'filename': filenames})

    train_df['label'] = train_df['filename'].apply(lambda x: int(x.split('.')[0].split('-')[-1]))
    train_df['filename'] = train_df['filename'].apply(lambda x: os.path.join(config.wav_train_path, x))
    train_df.to_csv(config.train_path, index=None)
