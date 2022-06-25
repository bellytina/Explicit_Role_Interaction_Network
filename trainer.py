import torch
from torch.autograd import Variable
from model import Toy_model
from transformers import BertConfig

from transformers import AdamW, get_linear_schedule_with_warmup

class Trainer(object):
    def __init__(self, args, embedding_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        raise NotImplementedError

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        #self.args = checkpoint['config']
        self.outer_config = checkpoint['config']

    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'config': self.outer_config #self.args,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch):
    for i in range(len(batch)):
        batch[i] = Variable(batch[i].cuda()) 
    return batch


class MyTrainer(Trainer):
    def __init__(self, outer_config):
        
        self.outer_config = outer_config
        
        bertconfig = BertConfig.from_pretrained(self.outer_config.transformers_path, num_labels=self.outer_config.tagset_size, task_specific_params=vars(self.outer_config))
        
        
        self.model = Toy_model.from_pretrained(
            self.outer_config.transformers_path, config=bertconfig).cuda()
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)], 'weight_decay': self.outer_config.weight_decay},
            {'params': [p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.outer_config.learning_rate, eps=self.outer_config.adam_epsilon)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.outer_config.warmup_steps, num_training_steps=self.outer_config.train_optimi_step)


    def update(self, batch, mode='train'):
        batch = unpack_batch(batch)
        # step forward
        self.model.train()
        loss = self.model(batch, mode)

        # backward of task loss
        self.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.outer_config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()  # Update learning rate schedule

        # loss value
        loss = loss.item()

        return loss 

    def predict(self, batch, mode='test'):
        with torch.no_grad():
            batch = unpack_batch(batch)
            # forward
            self.model.eval()
            pred, loss = self.model(batch, mode)

        return pred.tolist(), loss.item()
    
    def get_paras(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name,':',param.size())
        return


