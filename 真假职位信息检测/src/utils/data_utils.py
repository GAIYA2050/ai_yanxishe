# coding=utf-8
# author=yphacker


import torch
from torch.utils.data import Dataset
from conf import config


class MyDataset(Dataset):

    def __init__(self, df, tokenizer, mode='train'):
        self.mode = mode
        self.tokenizer = tokenizer
        self.columns = ['title', 'location', 'department', 'salary_range', 'company_profile', 'description',
                        'requirements', 'benefits', 'telecommuting', 'has_company_logo', 'has_questions',
                        'employment_type', 'required_experience', 'required_education', 'industry', 'function']
        self.x_data = []
        self.y_data = []
        for i, row in df.iterrows():
            x, y = self.row_to_tensor(self.tokenizer, row)
            self.x_data.append(x)
            self.y_data.append(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def solve(self,row):
        text = ''
        for column in self.columns:
            if column in ['company_profile', 'description', 'requirements', 'benefits']:
                text += (column + '' + str(row[column])[:100] + ' ')
            else:
                text += (column + '' + str(row[column]) + ' ')
        return text[:-1]

    def row_to_tensor(self, tokenizer, row):
        text_a = self.solve(row)
        # tokenizer.encode 自带截取功能
        inputs = tokenizer.encode_plus(text_a, max_length=config.max_seq_len, pad_to_max_length=True)
        y_tensor = torch.tensor(0, dtype=torch.long)
        if self.mode in ['train', 'val']:
            y_tensor = torch.tensor(row['fraudulent'], dtype=torch.long)

        # assert len(inputs["input_ids"]) == config.max_seq_len
        # assert len(inputs['attention_mask']) == config.max_seq_len
        # assert len(inputs["token_type_ids"]) == config.max_seq_len

        # print(inputs["input_ids"])
        # print(inputs["attention_mask"])
        # print(inputs["token_type_ids"])

        # 有些tokenizer.encode_plus返回不带token_type_ids
        x_tensor = torch.tensor(inputs["input_ids"], dtype=torch.long), \
                   torch.tensor(inputs['attention_mask'], dtype=torch.long), \
                   torch.tensor(inputs.get("token_type_ids", 0), dtype=torch.long)

        return x_tensor, y_tensor

    def __len__(self):
        return len(self.y_data)


if __name__ == "__main__":
    pass
