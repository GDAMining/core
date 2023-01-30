import os
import time
import math
import torch
import random
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, get_linear_schedule_with_warmup

from src.models import utils


parser = argparse.ArgumentParser()
parser.add_argument('--automatic_data', default='./data/disgenet/proc/automatic_data', type=str, help='DisGeNET automatic data to ingest.')
parser.add_argument('--curated_data', default='./data/disgenet/proc/curated_data', type=str, help='DisGeNET curated data to ingest.')
parser.add_argument('--manual_data', default='./data/manual/proc/manual_data', type=str, help='CECORE manual data to ingest.')
parser.add_argument('--bert', default='allenai/scibert_scivocab_uncased', type=str, help='BERT pre-trained model.')
parser.add_argument('--gpu', default=0, type=int, help='Primary GPU.')
parser.add_argument('--epochs', default=5, type=int, help='Number of max epochs.')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size.')
parser.add_argument('--warmup_frac', default=0.1, type=float, help='The fraction of iterations we perform warmup.')
parser.add_argument('--seed', default=42, type=int, help='Random seed.')

args = parser.parse_args()


def main():

    # Randomization

    # set seed value(s) to reproduce results
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Directories Creation

    # create model dir
    mdir = './trained/GCC_BERT'
    if not os.path.exists(mdir):
        os.makedirs(mdir)

    # GPU Setting

    # set GPU device
    gpu = args.gpu
    if torch.cuda.is_available():  # use GPU
        device = torch.device('cuda:' + str(gpu))
        print('GPU {} used: {}'.format(gpu, torch.cuda.get_device_name(gpu)))
    else:  # use CPU
        device = torch.device('cpu')
        print('GPUs not present, use CPU')

    # Data Processing

    print('Processing data ...')
    # create dataset for each data type -- i.e., automatic, curated, and manual -- and drop duplicates
    automatic_dataset = utils.create_dataset(args.automatic_data, 'GCC').drop_duplicates(subset=['Sentence', 'GCC'], ignore_index=True)
    curated_dataset = utils.create_dataset(args.curated_data, 'GCC').drop_duplicates(subset=['Sentence'], ignore_index=True)
    manual_dataset = utils.create_dataset(args.manual_data, 'GCC')

    # balance automatic dataset -- EXPRESSION > OTHER
    automatic_other = automatic_dataset[automatic_dataset['GCC'] == 0]
    automatic_expression = automatic_dataset[automatic_dataset['GCC'] == 1]
    # down-sample EXPRESSION to OTHER size
    automatic_expression = automatic_expression.sample(n=automatic_other.shape[0], random_state=args.seed)
    # combine both DataFrames back together
    automatic_dataset = pd.concat([automatic_other, automatic_expression])
    # shuffle DataFrame
    automatic_dataset = automatic_dataset.sample(frac=1).reset_index(drop=True)

    # remove sentences shared among datasets to avoid bias at inference time
    shared_sents = manual_dataset['Sentence']
    curated_dataset = curated_dataset[~curated_dataset['Sentence'].isin(shared_sents)]
    shared_sents = shared_sents.append(curated_dataset['Sentence'])
    automatic_dataset = automatic_dataset[~automatic_dataset['Sentence'].isin(shared_sents)]

    # create dataset to perform training and validation
    train_dataset = automatic_dataset
    valid_dataset = pd.concat([manual_dataset, curated_dataset])
    print('Data processed!')

    # BERT Tokenization

    # set BERT tokenizer
    print('setting BERT tokenizer ...')
    tokenizer = BertTokenizer.from_pretrained(args.bert)
    print('BERT tokenizer set!')

    # Training/Validation Split

    print('Preparing training and validation datasets ...')
    # prepare data loaders for training/validation/test
    train_data = utils.prepare_dataset(tokenizer, train_dataset['Sentence'], train_dataset['GCC'], batch_size=args.batch_size, train=True)
    valid_data = utils.prepare_dataset(tokenizer, valid_dataset['Sentence'], valid_dataset['GCC'], batch_size=1, train=False)
    print('Training and validation datasets prepared!')

    # Model Setting

    print('Setting BERT model ...')
    # set BERT
    config = BertConfig.from_pretrained(args.bert)
    config.output_attentions = False
    config.num_labels = 2
    config.label2id = utils.target2ix['GCC']
    config.id2label = {v: k for k, v in config.label2id.items()}

    model = BertForSequenceClassification.from_pretrained(args.bert, config=config)
    model = torch.nn.DataParallel(model)
    # set model to device
    model = model.to(device)

    # set the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # compute total number of training steps -- [number of batches] x [number of epochs]
    tot_steps = len(train_data) * args.epochs
    # compute number of warmup steps -- [tot_steps] * warmup_frac
    warmup_steps = math.floor(tot_steps * args.warmup_frac)

    # set scheduler for learning rate
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=tot_steps)
    print('BERT model set!')

    # Model Training

    # measure the total training time for the run
    total_tt = time.time()
    # set early stopping params
    best_f1 = 0
    best_epoch = 0

    # iterate over epochs
    for i in range(0, args.epochs):
        # perform a full pass over the training set
        print()
        print("======== Epoch {}/{} ========".format(i+1, args.epochs))
        print("Training")

        # start measuring training time
        epoch_tt = time.time()

        # reset total loss for this epoch
        total_train_loss = 0
        # set model into training mode
        model.train()

        # iterate over batches
        for step, batch in tqdm(enumerate(train_data), total=len(train_data)):
            # unpacking training batch from data loader and map each tensor to device
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_targets = batch[2].to(device)

            # perform forward pass
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_targets)
            # get training loss from outputs
            loss = outputs[0].mean()

            # accumulate training loss over batches
            total_train_loss += loss.item()

            # perform backward pass and update parameters
            loss.backward()
            # clip the norm of the gradients to 1.0 -- prevent "exploding gradients" problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # update parameters and take a step using the computed gradient
            optimizer.step()
            # update learning rate
            scheduler.step()

        # compute the average loss
        avg_train_loss = total_train_loss / len(train_data)
        # measure how long the epoch took
        train_time = utils.format_time(time.time() - epoch_tt)

        print()
        print("Average training loss: {}".format(avg_train_loss))
        print("Training epoch took: {}".format(train_time))

        # validation
        print()
        print("Validation")

        # put model in eval mode
        model.eval()

        # tracking variables
        total_valid_loss = 0

        # set validation predictions and target vars
        v_preds = []
        v_trues = []

        # validation for the current epoch
        for batch in valid_data:
            # unpack validation batch from dataloader and copy each tensor to GPU
            v_input_ids = batch[0].to(device)
            v_input_mask = batch[1].to(device)
            v_targets = batch[2].to(device)

            # avoid building the compute graph for validation
            with torch.no_grad():
                # perform a forward pass
                outputs = model(input_ids=v_input_ids, attention_mask=v_input_mask, labels=v_targets)
                # get validation loss from outputs
                loss = outputs[0]
                # get logits from outputs
                logits = outputs[1]

            # accumulate validation loss
            total_valid_loss += loss.item()

            # convert logits into predictions
            v_preds.append(torch.argmax(logits, dim=1).item())
            v_trues.append(v_targets.item())

        # compute the average loss
        avg_valid_loss = total_valid_loss / len(valid_data)

        print()
        print("Average validation loss: {}".format(avg_valid_loss))
        print()

        # convert validation prediction and target vars to numpy
        v_preds = np.array(v_preds).squeeze()
        v_trues = np.array(v_trues).squeeze()

        # compute metrics
        print('classification report for GCC:')
        print(classification_report(y_true=v_trues, y_pred=v_preds, target_names=[name for name in utils.target2ix['GCC'].keys()], digits=4))

        v_weighted_f1 = f1_score(y_true=v_trues, y_pred=v_preds, average='weighted')

        if v_weighted_f1 > best_f1:
            best_epoch = i + 1
            best_f1 = v_weighted_f1
            model.module.save_pretrained(mdir)

    print()
    print("Training complete -- best model found at epoch {}".format(best_epoch))
    print("Training took {} (h:mm:ss)".format(utils.format_time(time.time() - total_tt)))


if __name__ == "__main__":
    main()
