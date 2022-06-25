# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (BertModel, BertPreTrainedModel)

class Toy_model(BertPreTrainedModel):
    
    def __init__(self, config):
        super(BertPreTrainedModel, self).__init__(config)
        
        self.tagset_size  = config.num_labels
        self.dim = config.task_specific_params['hidden_dim']

        ###### bert encoder
        self.bert = BertModel(config)
        self.position_emb = nn.Embedding(len(config.task_specific_params['position2id']), config.hidden_size, padding_idx=0)
        self.event_emb    = nn.Embedding(len(config.task_specific_params['event2id']), config.hidden_size, padding_idx=0)
        self.BW = nn.Linear(config.hidden_size, self.dim) #scale output of bert
        ###### bert encoder
        
        
        ###### RIM
        self.unit_layer = config.task_specific_params['unit_layer']
        self.classifier = nn.Linear(self.dim, self.tagset_size)
        self.cla_drop = nn.Dropout(config.task_specific_params['cla_drop'])
        self.cell = nn.GRUCell(self.tagset_size*2, self.dim*2)
        
        self.TW = nn.Linear(self.dim*3, self.dim) #scale input of transformer
        self.T_input_drop = nn.Dropout(config.task_specific_params['trans_drop'])
        
        self.RIN = nn.ModuleList()
        for _ in range(self.unit_layer):
            self.RIN.append(InterUnit(config.task_specific_params)) 
        ###### RIM
        
        self.init_weights()
        self.ce_loss  = nn.CrossEntropyLoss(reduction='none')
        
  
    def forward(self, inputs, mode='train'):
        
        token1_ids, attn1_mask, segment1_ids, event1, position1, mask_r1, leng_r1, token2_ids, attn2_mask, segment2_ids, event2, position2, mask_r2, leng_r2, gather_index, mask_s, tags1, tags2, _ = inputs
        
        ##### role1
        # stc+event+position encoding
        word1_embs = self.bert.embeddings.word_embeddings(token1_ids)
        input1_embs = word1_embs+self.event_emb(event1)+self.position_emb(position1)
        bert1_outputs = self.bert(token_type_ids=segment1_ids, attention_mask=attn1_mask, inputs_embeds=input1_embs)
        sequence1_outputs = bert1_outputs[0] # sequence of hidden-states at the output of the last layer of the model 
        
        #split the text part and role part 
        lens = mask_s.sum(dim=1)
        
        pad_role1 = torch.zeros(segment1_ids.size(0), segment1_ids.size(1)-mask_s.size(1)).cuda() 
        mask_pad1 = torch.cat([mask_s, pad_role1], dim=-1)
        H1 = sequence1_outputs * mask_pad1.unsqueeze(-1)
        H1 = H1[:,:int(lens[0].item()),:] 
        
        # delte cls sep for text representation
        H1 = torch.gather(H1, 1, gather_index.unsqueeze(-1).repeat(1, 1, H1.size(-1)))
        H1 = F.relu(self.BW(H1)) 
        H1 = self.cla_drop(H1)
        Y1 = self.classifier(H1) 
        prob1 = F.softmax(Y1, dim=2)
        
        # avg pooling over role representation
        role1_outputs = sequence1_outputs*mask_r1.unsqueeze(-1)
        pooled1 = role1_outputs.sum(dim=1) / leng_r1.unsqueeze(-1) 
        role1 = F.relu(self.BW(pooled1)).unsqueeze(1).repeat(1, H1.size(1), 1)
        

        ##### role2
        word2_embs = self.bert.embeddings.word_embeddings(token2_ids)
        input2_embs = word2_embs+self.event_emb(event2)+self.position_emb(position2)
        bert2_outputs = self.bert(token_type_ids=segment2_ids, attention_mask=attn2_mask, inputs_embeds=input2_embs)
        sequence2_outputs = bert2_outputs[0] 
        
        pad_role2 = torch.zeros(segment2_ids.size(0), segment2_ids.size(1)-mask_s.size(1)).cuda() 
        mask_pad2 = torch.cat([mask_s, pad_role2], dim=-1)
        H2 = sequence2_outputs * mask_pad2.unsqueeze(-1)
        H2 = H2[:,:int(lens[0].item()),:]
        
        H2 = torch.gather(H2, 1, gather_index.unsqueeze(-1).repeat(1, 1, H2.size(-1)))
        H2 = F.relu(self.BW(H2)) 
        H2 = self.cla_drop(H2)
        Y2 = self.classifier(H2) 
        prob2 = F.softmax(Y2, dim=2)
        
        role2_outputs = sequence2_outputs*mask_r2.unsqueeze(-1)
        pooled2 = role2_outputs.sum(dim=1) / leng_r2.unsqueeze(-1) 
        role2 = F.relu(self.BW(pooled2)).unsqueeze(1).repeat(1, H2.size(1), 1)
        
        ##### Interaction
        role = torch.cat([role1,role2],dim=2) # gru input 
        Y = torch.cat([prob1,prob2],dim=2) # gru H0 
        Y = self.cell(Y.reshape(-1,self.tagset_size*2), role.reshape(-1, self.dim*2) ) # Nxdim
        Y = Y.resize(role.size(0),role.size(1), self.dim*2)

        for i in range(self.unit_layer):
            inputs1 = torch.cat([H1, Y],dim=2) 
            inputs1 = self.T_input_drop(inputs1)
            unit_inputs1 = self.TW(inputs1)
            H1 = self.RIN[i](unit_inputs1)
            Y1 = self.classifier(H1)
            prob1 = F.softmax(Y1, dim=2)
            
            inputs2 = torch.cat([H2, Y],dim=2)
            inputs2 = self.T_input_drop(inputs2)
            unit_inputs2 = self.TW(inputs2)
            H2 = self.RIN[i](unit_inputs2)
            Y2 = self.classifier(H2)
            prob2 = F.softmax(Y2, dim=2)
                        
            Y = torch.cat([prob1,prob2],dim=2) 
            Y = self.cell(Y.reshape(-1,self.tagset_size*2), role.reshape(-1, self.dim*2) ) 
            Y = Y.resize(role.size(0),role.size(1), self.dim*2)
        
        logits1 = Y1 
        logits2 = Y2  

        loss1 = self.ce_loss(logits1.reshape(-1, self.tagset_size), tags1.reshape(-1))
        loss1 = loss1.sum() / loss1.size(0)
        
        loss2 = self.ce_loss(logits2.reshape(-1, self.tagset_size), tags2.reshape(-1))
        loss2 = loss2.sum() / loss2.size(0)
        
        if mode == 'train':
            return loss1+loss2
        elif mode == 'test':
            pred_tag = torch.argmax(logits1, dim=2)
            return pred_tag, loss1+loss2
        

        
class InterUnit(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config['hidden_dim']
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=config['trans_head'])#embed_dim must be divisible by num_heads 512/8=64
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['trans_layer'])
         
    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)
        H = self.encoder(inputs)
        H = H.transpose(0, 1)
        return H

        




