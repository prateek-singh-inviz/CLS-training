from src.train.model.utils import *
from tqdm.auto import tqdm
tqdm.pandas()

class albert_multi_custom(torch.nn.Module):
    def __init__(self,output_len_4,pretrained_path = 'ecom_albert_nd_base',chan_len=10):
        super(albert_multi_custom, self).__init__()
        
        self.encoder = AlbertModel.from_pretrained(pretrained_path)
        

        self.L4_out_encoder = torch.nn.Conv1d(in_channels= output_len_4 , out_channels=chan_len, kernel_size=1)
        

        
        self.L4_drop = torch.nn.Dropout(0.3)
        self.L4_out = torch.nn.Linear(768, output_len_4)
        

        
    
    def forward(self, ids, mask, token_type_ids):
        
        _, embeddings= self.encoder(ids, attention_mask = mask, token_type_ids = token_type_ids,return_dict=False)
        
        L4_dropout = self.L4_drop(embeddings)
        L4_out = self.L4_out(L4_dropout)
        

        
        return L4_out
    def loss_fn(o, t):
        
        return torch.nn.BCEWithLogitsLoss()(o, t)

    def get_optimizer(self, model):
        optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,23,24,26,30], gamma=0.8)
        return optimizer, scheduler

def train(model,tokenizer,training_loader,epochs,epoch_start,save_path,load_checkpoint=True):
    joblib.dump(tokenizer,save_path+'/tokenizer.bin')
    loss_l=[9999,9999,9999]
    accuracy_l=[0]
    epoch_repo= {}
    m_=[]
    #!mkdir save_path
    if load_checkpoint:
        try:
            epoch_ = joblib.load(save_path+'/epoch.int')
            _id = joblib.load(save_path+'/id.int')
            model = joblib.load(save_path+'/model.bin')
        except:
            print('Initiating training from start') 
            
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
                joblib.dump(model,save_path+'/model.bin')
                
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

        joblib.dump(model,save_path+'/model_final_'+str(epoch)+'.bin')

            
    return model

