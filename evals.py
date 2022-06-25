# -*- coding: utf-8 -*-

# tag2id = {"B":1, "I":2, "O":0}
def split_arg(tags):
    ans = set()
    for i,t in enumerate(tags):
        if t==1: # B
            frm = i
            j = i+1
            while(j < len(tags)):
                if tags[j]!=2:
                    break
                j+=1
            to = j-1
            ans.add((frm, to))
    return ans


def evaluate_program(trainer, batches):

    # tag2id = {"B":1, "I":2, "O":0}
    # token1_ids, attn1_mask, segment1_ids, event1, position1, mask_r1, leng_r1, 
    # token2_ids, attn2_mask, segment2_ids, event2, position2, mask_r2, leng_r2,
    # mask_tag[:],  mask_s, #+2
    # tags1, tags2, flag

    
    golden_labels, pred_labels, lens, flags = [], [], [], []
    eval_loss, eval_step = 0., 0
    batch_num  = len(batches)
    
    for i in range(batch_num):
        batch = batches[i]
        pred, loss = trainer.predict(batch)
        
        eval_loss += loss
        eval_step += 1
        
        if eval_step % 10 == 0:
            print("eval_loss: {:.4f}".format(eval_loss/eval_step))

        mask_s = batch[-4]
        lens_ = mask_s.sum(dim=1) # includeing cls and sep
        
        golden_labels += batch[-3].tolist()
        pred_labels   += pred
        lens += lens_.tolist()
        flags += batch[-1].tolist()
        
    
    corrects = 0
    guesseds = 0
    golds    = 0

    # Loop over the data to compute a score
    for i in range(len(golden_labels)):
        l = int(lens[i])-2
        gold = golden_labels[i][:l]
        guess = pred_labels[i][:l]
        flag = flags[i]
        NO_ARG = [0]*l
        
         
        if gold == NO_ARG and guess == NO_ARG:
            pass
        
        elif gold != NO_ARG and guess == NO_ARG:
            golds += gold.count(1) # 'B'
            
        elif gold == NO_ARG and guess != NO_ARG:
            if flag==1: # ED is correct
                guesseds += guess.count(1)
        
        elif gold != NO_ARG and guess != NO_ARG:
            g_set = split_arg(gold)
            p_set = split_arg(guess)
            golds += len(g_set)
            if flag==1:
                guesseds += len(p_set)
                corrects += len(p_set & g_set)
    
    print("golds=",golds)
    print("guesseds=",guesseds)
    print("corrects=",corrects)

                
    prec_micro = 1.0
    if guesseds > 0:
        prec_micro   = float(corrects) / float(guesseds)
    recall_micro = 0.0
    if golds > 0:
        recall_micro = float(corrects) / float(golds)
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print( "Precision (micro): {:.3%}".format(prec_micro) )
    print( "   Recall (micro): {:.3%}".format(recall_micro) )
    print( "       F1 (micro): {:.3%}".format(f1_micro) )
    return prec_micro, recall_micro, f1_micro
    
    
    