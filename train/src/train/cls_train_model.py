import math
import os.path

import torch.nn.functional as nnf
from loguru import logger
from sklearn.metrics import multilabel_confusion_matrix, classification_report

from src.train.model.albert_ner import *
from src.train.model.utils import *

tqdm.pandas()


class cls_train_config:
    MAX_LEN = 50
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 16
    EPOCHS = 30
    LEARNING_RATE = 1e-05
    BASE_MODEL_PATH = './albert/'








def get_tokenized(self, text):
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=50,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        return {
            "ids": torch.tensor([ids], dtype=torch.long).to(
                self.device, dtype=torch.long
            ),
            "mask": torch.tensor([mask], dtype=torch.long).to(
                self.device, dtype=torch.long
            ),
            "token_type_ids": torch.tensor([token_type_ids], dtype=torch.long).to(
                self.device, dtype=torch.long
            ),
        }

def loss_fn(o, t):
        
    return torch.nn.BCEWithLogitsLoss()(o, t)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,23,24,26,30], gamma=0.8)


from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
from sklearn.preprocessing import normalize
import numpy as np

import itertools
from collections.abc import Iterable

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
    




def get_prediction(self, text, th):
        input_ = self.get_tokenized(text)
        out = self.model(input_["ids"], input_["mask"], input_["token_type_ids"])

        
        level4 = nnf.sigmoid(out).cpu().detach().numpy()[0]
        # levelb = out[4].cpu().detach().numpy()[0]

        level4 = self.get_sorted_filter_val(
            level4, th, self.mlb4)

        out = {
            "lower_level": {
                i: {"name": j["name"], "prob": j["prob"], "by": 1}
                for i, j in level4.items()
            },
        }

        return out



class CLSTrainModelJob():

    def __init__(self, args, config, s3_client) -> None:
        self.config = config
        self.args = args
        self.s3_client = s3_client

    @staticmethod
    def download_input_files(config, s3_client, tokenizer_local_path, data_local_path):

        # Download Combined train data csv file
        s3_client.download_file(config['source']['bucket_name'], config['source']['combined_train_data_path'],
                                data_local_path + config['source']['combined_train_data_path'].split('/')[-1])

        # Download Tokenizer files for Albert
        tokenizer_files = ["l4_ens_class.array", "tokenizer.json"]
        if not os.path.exists(f"{tokenizer_local_path}"):
            os.makedirs(f"{tokenizer_local_path}")
        for file_name in tokenizer_files:
            file_path = os.path.join(config['source']['cls_train_tokenizer'], file_name)
            local_file_path = tokenizer_local_path + file_name
            s3_client.download_file(config['source']['bucket_name'], file_path, local_file_path)
    @staticmethod
    def upload_output_files(output_path, config, s3_client):


        files = ["tokenizer.bin", "cls_albert_model.bin"]

        for file in files:
            s3_client.upload_file(output_path + file, config['sink']['bucket_name'],
                                  config['sink']['model_output_loc'] + os.path.basename(file))




    def train(model,tokenizer,training_loader,epochs,epoch_start,save_path,load_checkpoint=True):
        joblib.dump(tokenizer,save_path+'/tokenizer.bin')
        train_data = train_data.rename({'query':'text','category':'level4'},axis=1)

        train_data['level4'] = train_data['level4'].progress_apply(eval)
        mlb4=joblib.load(save_path+'/l4_ens_class.array')

        y4 = mlb4.transform(train_data['level4'])
        tokenizer = joblib.load(save_path+'/tokenizer.bin') ##while continue training
        x=list(train_data.text)

        MAX_LEN = 50
        TRAIN_BATCH_SIZE = 16
        VALID_BATCH_SIZE = 16
        EPOCHS = 30
        LEARNING_RATE = 1e-05
        loss_l=[9999,9999,9999]
        accuracy_l=[0]
        epoch_repo= {}
        m_=[]
        # !mkdir save_path
        training_set = CustomDataset(x,y4, tokenizer, MAX_LEN)
        train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

        test_params = {'batch_size': VALID_BATCH_SIZE,
                        'shuffle': True,
                        'num_workers': 0
                        }

        training_loader = DataLoader(training_set, **train_params)

        if load_checkpoint:
            try:
                epoch_ = joblib.load(save_path+'/epoch.int')
                _id = joblib.load(save_path+'/id.int')
                model = joblib.load(save_path+'/model.bin')
            except:
                logger.info('Initiating training from start') 
        logger.info('CLS Model Training Started')        
        for epoch in tqdm(range(epoch_start,epochs)):
            model.train()
            for _,data in tqdm(enumerate(training_loader, 0),total = len(training_loader)):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
    #             targets1 = data['targets1'].to(device, dtype = torch.float)
    #             targets2 = data['targets2'].to(device, dtype = torch.float)
    #             targets3 = data['targets3'].to(device, dtype = torch.float)
                targets4 = data['targets4'].to(device, dtype = torch.float)
    #             print(targets4.shape)
    #             targets4c = data['targetsb'].to(device, dtype = torch.float)
                outputs = model(ids, mask, token_type_ids)
    #             print(outputs.shape)
                optimizer.zero_grad()
    #             outputs = torch.cat((outputs[0],outputs[1],outputs[2],outputs[3],outputs[4]),dim=1)
    #             targets = torch.cat((targets1,targets2,targets3,targets4,targets4c),dim=1)
                loss = loss_fn(outputs, targets4)
                loss_l.append(loss)
                if _%3580==0:
                    if loss_l[-4] == loss:
                        scheduler.step()
                    print(f'Epoch: {epoch}, Loss:  {loss.item()}')
                    outputs = (np.array(outputs.cpu().detach().numpy()) > 0.5)*1
    #                 print(outputs)
                    targets4 = targets4.cpu().detach().numpy()
                    accuracy = metrics.accuracy_score(targets4, outputs)
                    if max(accuracy_l) < accuracy:
                        torch.save(model,save_path+'/model_best.bin')
                    print('\naccuracy = ',accuracy)
                    accuracy_l.append(accuracy)
                    m_.append(metrics_(targets4,outputs))
                    epoch_repo[epoch] = {'loss':loss_l,'accuracy':accuracy_l,'metrics':m_}
                    joblib.dump(epoch_repo,save_path+'/reports.dict')                
                    torch.save(model,save_path+'/model.bin')
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            logger.info('CLS Model Training Successful ')
  
            torch.save(model,'/media/HDD/aditya/model_files/'+'cls_albert_model'+'.bin')
        self.upload_output_files(output_path, config, s3_client)
        logger.info('Uploading Model Data Done!')
        return model
