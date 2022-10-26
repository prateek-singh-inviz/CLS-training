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
        LEARNING_RATE = 1e-05

        optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,23,24,26,30], gamma=0.8)
        return optimizer, scheduler

