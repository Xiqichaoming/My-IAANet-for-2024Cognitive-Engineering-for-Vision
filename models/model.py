from models.Attention import attention
from models.backbone import *
from models.transformer import Transformer

def get_model(device,trans_nel,trans_hidden_dim,pos_mode):
        
    Backbone = backbone()
    region_module = region_propose(Backbone)
    
    attention_module = Transformer(num_encoder_layers=trans_nel,d_model=trans_hidden_dim)
    
    model = attention(attention_module,region_module,pos=pos_mode,d_model=trans_hidden_dim)
    model.to(device)

    return model

        