import os
import time
import math
import torch
import random
import argparse
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AdamW, get_linear_schedule_with_warmup

from src.models import utils


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='./path/to/manual/data/file', type=str, help='Target data to ingest.')
parser.add_argument('--target', default='', type=str, help='Target factor to predict -- factors can be: "CGE", "CCS", "GCI".')
parser.add_argument('--num_labels', default=3, type=int, help='Number of classes to predict.')
parser.add_argument('--bert', default='allenai/scibert_scivocab_uncased', type=str, help='BERT pre-trained model.')
parser.add_argument('--valid_size', default=0.25, type=float, help='The proportion of the training set to include in the valid fold.')
parser.add_argument('--gpu', default=0, type=int, help='Primary GPU.')
parser.add_argument('--epochs', default=10, type=int, help='Number of max epochs.')
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
    mdir = './trained/' + args.target + '_BERT'
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

    # create dataset to perform training/validation/test
    dataset = utils.create_dataset(args.data, args.target)
    # convert dataset into inputs/targets format
    inputs = dataset['Sentence']
    targets = dataset[args.target]

    # BERT Tokenization

    # set BERT tokenizer
    print('set BERT tokenizer')
    tokenizer = BertTokenizer.from_pretrained(args.bert)

    # Training/Validation Preparation

    # divide dataset intro training and validation
    train_inputs, valid_inputs, train_targets, valid_targets = train_test_split(inputs, targets, test_size=args.valid_size, random_state=args.seed, shuffle=True, stratify=targets)
    # prepare data loaders for training/validation/test
    train_data = utils.prepare_dataset(tokenizer, train_inputs, train_targets, batch_size=args.batch_size, train=True)
    valid_data = utils.prepare_dataset(tokenizer, valid_inputs, valid_targets, batch_size=1, train=False)

    # Model Setting

    # set BERT
    config = BertConfig.from_pretrained(args.bert)
    config.output_attentions = False
    config.num_labels = args.num_labels
    config.label2id = utils.target2ix[args.target]
    config.id2label = {v: k for k, v in config.label2id.items()}

    model = BertForSequenceClassification.from_pretrained(args.bert, config=config)
    model = torch.nn.DataParallel(model)
    # set model to device
    model = model.to(device)

    # set the optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # compute total number of training steps -- [number of batches] x [number of epochs]
    tot_steps = len(train_data) * args.epochs
    # compute number of warmup steps -- [tot_steps] * warmup_frac
    warmup_steps = math.floor(tot_steps * args.warmup_frac)

    # set scheduler for learning rate
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=tot_steps)

    # Model Training

    # measure the total training time for the run
    total_tt = time.time()
    # set early stopping params
    best_epoch = 0
    best_f1 = 0

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

        # convert validation prediction and target vars to numpy
        v_preds = np.array(v_preds).squeeze()
        v_trues = np.array(v_trues).squeeze()

        # compute metrics
        print('classification report for {}:'.format(args.target))
        print(classification_report(y_true=v_trues, y_pred=v_preds, target_names=[name for name in utils.target2ix[args.target].keys()]))

        v_weighted_f1 = f1_score(y_true=v_trues, y_pred=v_preds, average='weighted')

        if v_weighted_f1 > best_f1:
            # store trained model
            model.module.save_pretrained(mdir)
            # update best params and patience
            best_epoch = i + 1
            best_f1 = v_weighted_f1

    print()
    print("Training complete -- best model found at epoch {}".format(best_epoch))
    print("Training took {} (h:mm:ss)".format(utils.format_time(time.time() - total_tt)))


if __name__ == "__main__":
    main()
