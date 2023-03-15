
import os
import argparse
from logger import Logger

from lhotse.dataset.collation import TokenCollater
from lhotse.dataset import DynamicBucketingSampler
from lhotse.dataset import make_worker_init_fn
from lhotse.dataset.iterable_dataset import IterableDatasetWrapper
import editdistance
from CTC_loss_function import CTCLoss
from ASRDataset_class import ASRDataset
from Hubert_Encoder import LSTMEncoder
from Conformer_Encoder import SpeechEncoder
import get_data_function
from checkpoint_functions import save_checkpoint, load_checkpoint
import torch
import torch.nn as nn

import numpy
import time

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--main-dir",
        type=str,
        default="/home/eorenst1/pipeline_hubert",
        help="main directory of the process",
    )
    parser.add_argument(
        "--running-name",
        type=str,
        default="basic_running",
        help="name of specific runnig, the logs and the checkpint will save by this name",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="how many epochs will train",
    )
    parser.add_argument(
        "--first-epoch",
        type=int,
        default=0,
        help="how many epochs will train",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=140.0,
        help="for 2 gpu. if there is OOM, reduce this number",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        help="the erchitecure of the model",
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    name_of_running = args.running_name
    main_dir = args.main_dir
    print(f"start {name_of_running} pipline")
    # Some global variables
    download = True
    data = None # /path/to/
    num_epochs = args.num_epochs
    first_epoch = args.first_epoch
    
    log_dir = main_dir + "/log"
    name_dir = "log_"+name_of_running
    log_full_path = os.path.join(log_dir, name_dir)
    if(not os.path.isdir(log_full_path)):
        os.mkdir(log_full_path)
    logger = Logger(log_full_path)
    logger.writer.flush()

    checkpoint_dir = main_dir + "/checkpoint"
    specific_checkpoint_iter_dic = "cp_"+name_of_running
    cp_full_path = os.path.join(checkpoint_dir, specific_checkpoint_iter_dic)
    if(not os.path.isdir(cp_full_path)):
        os.mkdir(cp_full_path)


    # Some LR stuff
    min_lr = 1e-07
    max_lr = 1e-05
    warmup = 8000 # In Hubert paper they use 8k, and 80k steps total
    freeze = 10000 # In Hubert paper they use 10k
    lr_slope = (max_lr - min_lr) / warmup # The schedule is slightly different in HUBERT
    

    cuts_dir = main_dir + '/data/cuts' # /path/to/

    print("get data function")
    # Get the data
    cuts_train, cuts_dev, cuts_test = get_data_function.get_data(cuts_dir)
    cuts_train.describe()
    # Get the text tokenizer
    tokenizer = TokenCollater(cuts_train)

    # Define the dataset, samplers and data loaders.
    # These are responsible for batching the data during nnet training
    train_dataset = ASRDataset(tokenizer)
    dev_dataset = ASRDataset(tokenizer)
    train_sampler = DynamicBucketingSampler(
        cuts_train,
        max_duration=args.max_duration,
        shuffle=True,
        num_buckets=100,
    )
    dev_sampler = DynamicBucketingSampler(
        cuts_dev,
        max_duration=args.max_duration,
        shuffle=False,
    )
    #train_iter_dataset = IterableDatasetWrapper(
    #dataset=train_dataset,
    #sampler=train_sampler,
    #)
    train_dloader = torch.utils.data.DataLoader(
    train_dataset, sampler=train_sampler, batch_size=None, num_workers=4,
    )
    dev_dloader = torch.utils.data.DataLoader(
        dev_dataset, sampler=dev_sampler, batch_size=None, num_workers=4
    )
    print ("dev information")
    count_data = len(list(cuts_dev.data.items()))
    print (count_data)
    print('the first item: {}', list(cuts_dev.data.items())[0])

    # Create the model
    if(args.model=='lstm'):
        model = LSTMEncoder(len(tokenizer.idx2token), freeze_updates=freeze)
    if(args.model=='conformer'):
        model = SpeechEncoder(len(tokenizer.idx2token), freeze_updates=freeze)

    #print the count of trainig params:
    print ( "the count of training params:") 
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # Create the loss function
    loss_fn = CTCLoss()
    # Create the optimizer
    # Normally we would warmup the learning rate and then decay it, but for
    # simplicity we are just using a small fixed rate
    optim = torch.optim.Adam(
        list(filter(lambda p: p.requires_grad, model.parameters())),
        lr=min_lr,
        weight_decay=1e-06,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        
    else:
        device = torch.device("cpu")
    print(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        #optim = nn.DataParallel(optim)
        
    model.to(device)
    loss_fn.to(device)

    
    if first_epoch>0:
        path_checkpoint = os.path.join(cp_full_path, f"checkpoint_{first_epoch-1}.pt")
        checkpoint = load_checkpoint(path_checkpoint)
        #model.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(torch.load(path_checkpoint), strict=False)
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        print ("first epoch from the checkpoint: " + str(first_epoch))
        loss = checkpoint['loss']

    scaler = torch.cuda.amp.GradScaler()
    iter_num = 0
    curr_lr = min_lr
    # Looping over epochs
    for e in range(first_epoch, num_epochs):
        num_batches = sum(1 for b in train_dloader.sampler)
        # Looping over minibatches within an epoch
        for batch_idx, b in enumerate(train_dloader):
            batch_size = b['input'].size(0)
            # Move the minibatch to CUDA
            b = ASRDataset.move_to(b, device)
            with torch.cuda.amp.autocast():
                # Pass minibatch inputs through the encoder and output linear layer.
                # Remember that iter_num is used internally to decide
                # whether to freeze underlying encoder or not 
                #outputs = model(b, iter_num=iter_num)
                outputs = model(**dict(mb = b, iter_num=iter_num))
                loss = loss_fn(outputs, b['target'])
            # I don't think we need these, but in case something goes wrong, we are
            # just skipping these batches
            if loss is None:
                del b
                del loss
                continue;
            elif loss.isinf() or loss.isnan():
                print("NaN or Inf loss. Skipping ...")
                del b
                del loss
                continue;
            loss = loss.sum() / batch_size
            scaler.scale(loss).backward()
            loss.detach()
            scaler.unscale_(optim)
            params = model.parameters()
            grad_norm = nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

           
            # Print some statistics occasionally
            if batch_idx % 50 == 0:
                print(
                    'Iter: {:d}/{:d} Loss: {:.04f}'.format(
                        batch_idx, num_batches, loss.data.item()
                    )
                )

            iter_num += 1
            
            # Update the learning rate
            curr_lr = curr_lr + lr_slope if iter_num <= warmup else curr_lr - lr_slope
            #curr_lr = min_lr
            for param_group in optim.param_groups:
                param_group['lr'] = curr_lr
            # ============ TensorBoard logging ============#
            # (1) Log the scalar values
            info = {
                'train/loss': loss.detach().cpu(),
                'train/lr': curr_lr,
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, iter_num)
        
        print ("START VALIDATION")
        # Validation
        loss_val = 0.0
        model.eval()
        num_val = 0.0
        errors, num_ref = 0, 0
        examples = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            print ("get in to with")
            for batch_idx, b in enumerate(dev_dloader):
                if (batch_idx%50 ==0):
                    print ("batch num: " + str(batch_idx))
                b = ASRDataset.move_to(b, device)
                outputs = model(b)
                loss_val += loss_fn(outputs, b['target']).sum()
                preds = outputs[0].argmax(-1)
                preds2 = [preds[i].unique_consecutive()[preds[i].unique_consecutive() != 0] for i in range(preds.size(0))]  
                lens =  [preds2[i].size(0) for i in range(preds.size(0))]
                output_text = tokenizer.inverse(preds2, lens)
                if batch_idx % 10 == 0:
                    examples.append((output_text[0], b['text'][0]))
                for i, text in enumerate(output_text):
                    errors += editdistance.eval("".join(text).split(), b['text'][i].split())
                    num_ref += len(b['text'][i].split())
                    num_val += 1.0
            wer_str = "WER: {:.02f}".format(100.0 * errors / num_ref)
            for eg in examples:
                print("hyp: ", eg[0])
                print("ref: ", eg[1])
            print(wer_str)
            
            
            model.train()
            loss_val /= num_val
            print("---------------------------")
            print("Epoch {:d} Loss: {:.04f} {}".format(e+1, loss_val, wer_str))
            # ============ TensorBoard logging ============#
            # (1) Log the scalar values
            info = {
                'train/loss_val': loss_val.detach().cpu(),
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, iter_num)
        save_checkpoint(e, model, optim, loss_val, cp_full_path)
        print ("end epoch - after with")


if __name__ == "__main__":
    main()