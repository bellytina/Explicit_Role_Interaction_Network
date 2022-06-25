# -*- coding: utf-8 -*-

import json
import random
import torch
import numpy as np
from transformers import BertTokenizer
from random import choice

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, config, file_path, evaluation=False):
        self.batch_size = config.per_gpu_train_batch_size if evaluation==False else config.per_gpu_eval_batch_size
        self.eval = evaluation
        self.file_path = file_path
        
        do_lower_case = "-uncased" in config.pretrained_model_name
        self.tokenizer = BertTokenizer.from_pretrained("%s/vocab.txt"  % (config.transformers_path), do_lower_case=do_lower_case) 
 
        with open(file_path, 'r') as f:
            self.raw_data = json.load(f)
        self.position2id, self.tag2id, self.rel2id,self.role2id, self.event2role, self.event2id = config.position2id, config.tag2id, config.rel2id, config.role2id, config.event2role, config.event2id
        self.data = self.preprocess(self.raw_data)

        self.num_examples = len(self.data)

        # chunk into batches
        self.data = [self.data[i:i+self.batch_size] for i in range(0, len(self.data), self.batch_size)]
        print("{} examples are divided into {} batches created for {}".format(self.num_examples, len(self.data), self.file_path))


    def preprocess(self, data):

        processed = []
        
        for d in data:
            #"event_type": "Personnel:Nominate", "from": 9,"to": 9,
            r_tokens = d['tokens'] #
            
            
            subwords_2d = list(map(self.tokenizer.tokenize, r_tokens))
            subword_lengths = list(map(len, subwords_2d))      
            start_idx = np.cumsum([0]+subword_lengths[:-1]) 
            
            tokens = sum(subwords_2d,[]) 

            leng = len(tokens)

            mask_tag = [1 for i in range(leng+2)] 
            mask_tag[0], mask_tag[-1] = 0,0 # used to delete the cls and sep
            
            if self.eval:# predict trigger for inference
                t_from, t_to= d['from'], d['to']
            else:# golden trigger for training
                t_from, t_to= start_idx[d['from']],start_idx[d['to']]+subword_lengths[d['to']]-1   
            
            #additional input: position and event
            position = [i-t_from for i in range(-1, t_from)] + [0 for _ in range(t_from, t_to+1)] + [i-t_to for i in range(t_to+1, leng+1)]
            position = map_to_ids(position, self.position2id)
            
            subevent = d['event_type']
            event = [self.event2id[subevent] for i in range(leng+2)] 
            
            token_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]']+tokens+['[SEP]'])
            segment_ids = [0 for i in range(len(token_ids))]
            attn_mask = [1 for i in range(len(token_ids))]
            
            mask_s = [1 for i in range(leng+2)] # mask for "text+cls+sep"
            mask_r = [0 for i in range(leng+2)] # mask for "role"
            
            assert len(position)-2 == len(event)-2 == len(tokens) 
            
            flag=d['flag'] 
            
            seen_roles = {} 
            all_roles = self.event2role[subevent] 
            
            for arg in d['args']:# "role": "Person", "entity_type": "PER:Individual", "start": 10, "end": 18)
                role = arg['role']
                a_start, a_end = start_idx[arg['start']],start_idx[arg['end']]+subword_lengths[arg['end']]-1
                if role not in seen_roles:
                    seen_roles[role] = set() 
                seen_roles[role].add((a_start, a_end)) 
            
            for role1 in seen_roles:
                while(True):
                    role2 = choice(all_roles)
                    if role2 != role1:break
                    
                tags1 = ['O' for i in range(leng)]
                for start, end in seen_roles[role1]:
                    tags1[start] = 'B'
                    for i in range (start+1, end+1):
                       tags1[i] = 'I'
                tags1 = [self.tag2id[t] for t in tags1]
                
                role1_tokens = self.tokenizer.tokenize(role1)
                leng_r1 = len(role1_tokens)
                token1_ids = token_ids + self.tokenizer.convert_tokens_to_ids(role1_tokens + ['[SEP]'])
                segment1_ids = segment_ids + [1 for i in range(leng_r1+1)]
                attn1_mask = attn_mask + [1 for i in range(leng_r1+1)]
                position1 = position + [0 for i in range(leng_r1+1)]
                event1 = event + [0 for i in range(leng_r1+1)]
                mask_r1 = mask_r + [1 for i in range(leng_r1)] + [0] 
                #assert len(token1_ids) == len(segment1_ids) == len(attn1_mask) == len(position1) == len(event1) == len(mask_r1)
                
                
                tags2 = ['O' for i in range(leng)]
                if role2 in seen_roles:
                    for start, end in seen_roles[role2]:
                        tags2[start] = 'B'
                        for i in range (start+1, end+1):
                           tags2[i] = 'I'          
                tags2 = [self.tag2id[t] for t in tags2]

                role2_tokens = self.tokenizer.tokenize(role2)
                leng_r2 = len(role2_tokens)
                token2_ids = token_ids + self.tokenizer.convert_tokens_to_ids(role2_tokens + ['[SEP]'])
                segment2_ids = segment_ids + [1 for i in range(leng_r2+1)]
                attn2_mask = attn_mask + [1 for i in range(leng_r2+1)]
                position2 = position + [0 for i in range(leng_r2+1)]
                event2 = event + [0 for i in range(leng_r2+1)]
                mask_r2 = mask_r + [1 for i in range(leng_r2)] + [0]
                
                       
                processed.append(
                    [token1_ids, attn1_mask, segment1_ids, event1, position1, mask_r1, leng_r1, # cls txet sep role sep
                     token2_ids, attn2_mask, segment2_ids, event2, position2, mask_r2, leng_r2,# cls txet sep role sep
                     mask_tag[:],  mask_s, #+2
                     tags1, tags2, flag]) 

            for role1 in all_roles:
                if role1 not in seen_roles:
                    if self.eval == True or self.eval == False and random.random()>0.5:
                        while(True):
                            role2 = choice(all_roles)
                            if role2 != role1:break
                            
                        tags1 = ['O' for i in range(leng)]
                        tags1 = [self.tag2id[t] for t in tags1]

                        role1_tokens = self.tokenizer.tokenize(role1)
                        leng_r1 = len(role1_tokens)
                        token1_ids = token_ids + self.tokenizer.convert_tokens_to_ids(role1_tokens + ['[SEP]'])
                        segment1_ids = segment_ids + [1 for i in range(leng_r1+1)]
                        attn1_mask = attn_mask + [1 for i in range(leng_r1+1)]
                        position1 = position + [0 for i in range(leng_r1+1)]
                        event1 = event + [0 for i in range(leng_r1+1)]
                        mask_r1 = mask_r + [1 for i in range(leng_r1)] + [0]
                        
                        tags2 = ['O' for i in range(leng)]
                        if role2 in seen_roles:
                            for start, end in seen_roles[role2]:
                                tags2[start] = 'B'
                                for i in range (start+1, end+1):
                                   tags2[i] = 'I'               
                        tags2 = [self.tag2id[t] for t in tags2]

                        role2_tokens = self.tokenizer.tokenize(role2)
                        leng_r2 = len(role2_tokens)
                        token2_ids = token_ids + self.tokenizer.convert_tokens_to_ids(role2_tokens + ['[SEP]'])
                        segment2_ids = segment_ids + [1 for i in range(leng_r2+1)]
                        attn2_mask = attn_mask + [1 for i in range(leng_r2+1)]
                        position2 = position + [0 for i in range(leng_r2+1)]
                        event2 = event + [0 for i in range(leng_r2+1)]
                        mask_r2 = mask_r + [1 for i in range(leng_r2)] + [0]
       
                        processed.append(
[token1_ids, attn1_mask, segment1_ids, event1, position1, mask_r1, leng_r1, token2_ids, attn2_mask, segment2_ids, event2, position2, mask_r2, leng_r2, mask_tag[:], mask_s, tags1, tags2, flag]) 
    
                       

        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        # token1_ids, attn1_mask, segment1_ids, event1, position1, mask_r1, leng_r1, 
        # token2_ids, attn2_mask, segment2_ids, event2, position2, mask_r2, leng_r2,
        # mask_tag[:],  mask_s, #+2
        # tags1, tags2, flag
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 19

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[15]] #mask_s
        batch, _ = sort_all(batch, lens)

        # convert to tensors
        token1_ids   = get_long_tensor(batch[0], batch_size)#words
        attn1_mask   = get_long_tensor(batch[1], batch_size)
        segment1_ids = get_long_tensor(batch[2], batch_size)
        event1       = get_long_tensor(batch[3], batch_size)
        position1    = get_long_tensor(batch[4], batch_size)
        mask_r1      = get_float_tensor(batch[5], batch_size)
        leng_r1      = torch.tensor(batch[6]).float()
        
        token2_ids   = get_long_tensor(batch[7], batch_size)#words
        attn2_mask   = get_long_tensor(batch[8], batch_size)
        segment2_ids = get_long_tensor(batch[9], batch_size)
        event2       = get_long_tensor(batch[10], batch_size)
        position2    = get_long_tensor(batch[11], batch_size)
        mask_r2      = get_float_tensor(batch[12], batch_size)
        leng_r2      = torch.tensor(batch[13]).float()

        gather_index = get_gather_tensor(batch[14], batch_size)
        
        mask_s   = get_float_tensor(batch[15], batch_size)
        
        tags1    = get_long_tensor(batch[16], batch_size)
        tags2    = get_long_tensor(batch[17], batch_size)
        flag     = torch.tensor(batch[-1]).long() 
        
        assert mask_s.size(1)-2 == gather_index.size(1)   
        
        return [token1_ids, attn1_mask, segment1_ids, event1, position1, mask_r1, leng_r1, token2_ids, attn2_mask, segment2_ids, event2, position2, mask_r2, leng_r2, gather_index, mask_s, tags1, tags2, flag] 



    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [1 if x != 1 and np.random.random() < dropout else x for x in tokens]
            
            
def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else vocab['[UNK]'] for t in tokens]
    return ids

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def get_float_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded FloatTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.FloatTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]
    
def get_gather_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to gather index tensor. """
    token_len = max(len(x) for x in tokens_list)
    for x in tokens_list:
        x += (token_len-len(x))*[1]
    gather_index = []
    for x in tokens_list:
        gather_index.append([i for i in range(token_len) if x[i]!=0])
    return torch.tensor(gather_index, dtype=torch.long)
