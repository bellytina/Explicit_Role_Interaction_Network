# -*- coding: utf-8 -*-
import os
import random
from config import Config
from argparse import ArgumentParser
from evals import evaluate_program
from loader import DataLoader 
from trainer import MyTrainer

seed = random.randint(1, 10000)

class FileLogger(object):
    """
    A file logger that opens the file periodically and write to it.
    """
    def __init__(self, filename, header=None):
        self.filename = filename
        if os.path.exists(filename):
            # remove the old file
            os.remove(filename)
        if header is not None:
            with open(filename, 'w') as out:
                print(header, file=out)
    
    def log(self, message):
        with open(self.filename, 'a') as out:
            print(message, file=out)

parser = ArgumentParser(description="BERT for argument extraction")
parser.add_argument('--config', default='config.ini')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')

args = parser.parse_args()
config = Config(args.config)

# load vocab and embedding matrix
dataset_path = "./data/%s"  % (config.dataset)

# load data
train_path  = '%s/train.json' % (dataset_path)
dev_path    = '%s/dev.json'   % (dataset_path)
test_path   = '%s/test.json'  % (dataset_path)


print("Loading data from {} ...".format(dataset_path))
train_batches  = DataLoader(config, train_path, evaluation=False)
dev_batches    = DataLoader(config, dev_path, evaluation=True)
test_batches   = DataLoader(config, test_path, evaluation=True)

config.train_optimi_step = len(train_batches) * config.num_train_epochs

# create the folder for saving the best model
if os.path.exists(args.save_dir) != True:
    os.mkdir(args.save_dir)

log_file = FileLogger(args.save_dir+"/log.txt")

print('Building model...')
# create model
trainer_model  = MyTrainer(config)

trainer_model.get_paras()

# start training
batch_num  = len(train_batches)
current_best_f1 = -1 
estop = 0
for epoch in range(1, int(config.num_train_epochs)+1):
    
    if estop >=config.early_stop:
        break

    train_loss, train_step = 0., 0
    for i in range(batch_num):
        batch = train_batches[i]
        loss = trainer_model.update(batch)
        train_loss += loss
        train_step += 1
        
        # print training loss 
        if train_step % config.print_step == 0:
            print("[{}] train_loss: {:.4f}".format(epoch, train_loss/train_step))
    
    # evaluate on unlabel set
    print("")
    print("Evaluating...Epoch: {}".format(epoch))
    pre, rec, f1 = evaluate_program(trainer_model, dev_batches)
    print("f1: {:.4f}".format(f1))
    # loging
    log_file.log("f1: {:.4f}".format(f1))

    if f1 > current_best_f1:
        current_best_f1 = f1
        trainer_model.save(args.save_dir+'/best_model.pt')
        print("New best model saved!")
        log_file.log("New best model saved!")
        estop = 0

    else:
        estop += 1
    print("")


print("Training ended with {} epochs.".format(epoch))

# final results
trainer_model.load(args.save_dir+'/best_model.pt')
pre, rec, f1 = evaluate_program(trainer_model, test_batches)

print("Final result:")
print("f1: {:.4f}".format(f1))

# loging
log_file.log("Final result:")
log_file.log("f1: {:.4f}".format(f1))

of = open('tmp.txt','a')
of.write("{:.2f}".format(pre*100)+'\t'+"{:.2f}".format(rec*100)+'\t'+"{:.2f}".format(f1*100)+'\n')
of.close()  