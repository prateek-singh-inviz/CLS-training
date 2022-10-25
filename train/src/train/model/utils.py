import torch
import torch.nn as nn
from transformers import AlbertTokenizerFast, AlbertModel, AlbertConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
from sklearn.preprocessing import normalize
import numpy as np
import itertools
from collections.abc import Iterable
import joblib
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

#from tqdm.auto import tqdm
#tqdm.pandas()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    
    def __init__(self, x,y4, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = x
#         self.targets1 = y1
#         self.targets2 = y2
#         self.targets3 = y3
        self.targets4 = y4
#         self.targetsb = yb
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
#             'targets1': torch.tensor(self.targets1[index], dtype=torch.float),
#             'targets2': torch.tensor(self.targets2[index], dtype=torch.float),
#             'targets3': torch.tensor(self.targets3[index], dtype=torch.float),
            'targets4': torch.tensor(self.targets4[index], dtype=torch.float),
#             'targetsb': torch.tensor(self.targetsb[index], dtype=torch.float)
        }
# class InferenceDataset:
    def __init__(self, texts,  config,TOKENIZER):
        self.texts = texts
        self.config = config
        self.TOKENIZER = TOKENIZER

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]

        ids = []

        for i, s in enumerate(text):
            inputs = self.TOKENIZER.encode(
                s,
                add_special_tokens=False
            )
            input_len = len(inputs)
            ids.extend(inputs)

        ids = ids[:self.config.MAX_LEN - 2]

        ids = [2] + ids + [3]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = self.config.MAX_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long)
        }
    


def flatten(data):
    for x in data:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

def metrics_(y,y_):
    
    y_true = np.array(list(flatten(y)))
    y_pred = np.array(list(flatten(y_)))

    y_indexs = set(list(np.where(y_true == 1)[0]) + list(np.where(y_pred > 0.5 )[0]))
    
    y_true = [y_true[i] for i in y_indexs]
    y_pred = [y_pred[i] for i in y_indexs]
    
    print('batch recall = ',recall_score(y_true, y_pred, average='micro'))
    print('batch precision = ',precision_score(y_true, y_pred, average='micro'))
    print('batch F1 = ',f1_score(y, y_, average='micro'))
    print('batch F1 flat = ',f1_score(y_true, y_pred, average='micro'))
    print('batch F1 macro = ',f1_score(y, y_, average='macro'))
    return {
        'batch recall':recall_score(y_true, y_pred, average='micro'),
        'batch precision':precision_score(y_true, y_pred, average='micro'),
        'batch F1':f1_score(y, y_, average='micro'),
        'batch F1 flat':f1_score(y_true, y_pred, average='micro'),
        'batch F1 macro':f1_score(y, y_, average='macro')
    }    
    
