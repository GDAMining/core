import os
import time
import math
import torch
import random
import argparse
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AdamW, get_linear_schedule_with_warmup

from src.models import utils


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='./path/to/manual/data/file', type=str, help='Target data to ingest.')
parser.add_argument('--target', default='', type=str, help='Target factor to predict -- factors can be: "CGE", "CCS", "GCI".')
parser.add_argument('--num_labels', default=3, type=int, help='Number of classes to predict.')
parser.add_argument('--num_folds', default=10, type=int, help='Number of folds used to perform stratified k-fold cv.')
parser.add_argument('--bert', default='allenai/scibert_scivocab_uncased', type=str, help='BERT pre-trained model.')
parser.add_argument('--valid_size', default=0.25, type=float, help='The proportion of the training set to include in the valid fold')
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
    mdir = './evaluated/' + args.target + '_BERT'
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

    # Stratified K-Fold Cross Validation

    # set stratified k-fold cv
    kfold = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

    # set CGE, CCS, and PT fold vars
    fold_epoch = []
    fold_acc = []
    fold_micro_prec = []
    fold_micro_rec = []
    fold_micro_f1 = []
    fold_weighted_prec = []
    fold_weighted_rec = []
    fold_weighted_f1 = []
    fold_num = 0

    for train, test in kfold.split(inputs, targets):
        # set starting fold number to one
        fold_num += 1
        print('########## Fold {} ##########\n'.format(fold_num))

        # Training/Validation/Test Preparation

        # divide training into training + validation
        train_inputs, valid_inputs, train_targets, valid_targets = train_test_split(inputs.iloc[train], targets.iloc[train], test_size=args.valid_size, random_state=args.seed, shuffle=True, stratify=targets.iloc[train])
        # prepare data loaders for training/validation/test
        train_data = utils.prepare_dataset(tokenizer, train_inputs, train_targets, batch_size=args.batch_size, train=True)
        valid_data = utils.prepare_dataset(tokenizer, valid_inputs, valid_targets, batch_size=1, train=False)
        test_data = utils.prepare_dataset(tokenizer, inputs.iloc[test], targets.iloc[test], batch_size=1, train=False)

        # Model Setting

        # set BERT
        config = BertConfig.from_pretrained(args.bert)
        config.num_labels = args.num_labels
        config.output_attentions = False

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
            print('classification report for {}:'.format(args.target))
            print(classification_report(y_true=v_trues, y_pred=v_preds, target_names=[name for name in utils.target2ix[args.target].keys()], digits=4))

            v_weighted_f1 = f1_score(y_true=v_trues, y_pred=v_preds, average='weighted')

            if v_weighted_f1 > best_f1:
                best_epoch = i + 1
                best_f1 = v_weighted_f1
                model.module.save_pretrained(mdir)

        # store best epoch for the current fold
        fold_epoch.append(best_epoch)

        print()
        print("Training complete -- best model found at epoch {}".format(best_epoch))
        print("Training took {} (h:mm:ss)".format(utils.format_time(time.time() - total_tt)))

        # delete model and release GPU memory before testing
        model = model.to('cpu')
        torch.cuda.empty_cache()
        del model

        print("Predicting targets for {} test sentences".format(inputs.iloc[test].shape[0]))

        # load saved model
        model = BertForSequenceClassification.from_pretrained(mdir)
        model = model.to(device)

        # put model in eval mode
        model.eval()

        # set test predictions and target vars
        t_preds = []
        t_trues = []

        # predict
        for batch in test_data:
            # unpack inputs from dataloader
            t_input_ids = batch[0].to(device)
            t_input_mask = batch[1].to(device)
            t_targets = batch[2].to(device)

            # avoid building the compute graph during testing
            with torch.no_grad():
                outputs = model(input_ids=t_input_ids, attention_mask=t_input_mask)

            # convert logits into predictions
            t_preds.append(torch.argmax(outputs[0], dim=1).item())
            t_trues.append(t_targets.item())

        # convert validation prediction and target vars to numpy
        t_preds = np.array(t_preds).squeeze()
        t_trues = np.array(t_trues).squeeze()

        # compute metrics
        print('classification report for {}:'.format(args.target))
        print(classification_report(y_true=t_trues, y_pred=t_preds, target_names=[name for name in utils.target2ix[args.target].keys()], digits=4))

        # compute test accuracy and weighted precision, recall, and f1 scores
        t_acc = accuracy_score(y_true=t_trues, y_pred=t_preds)
        t_weighted_prec = precision_score(y_true=t_trues, y_pred=t_preds, average='weighted')
        t_weighted_rec = recall_score(y_true=t_trues, y_pred=t_preds, average='weighted')
        t_weighted_f1 = f1_score(y_true=t_trues, y_pred=t_preds, average='weighted')

        # store test measures for the current fold
        fold_acc.append(t_acc)
        fold_weighted_prec.append(t_weighted_prec)
        fold_weighted_rec.append(t_weighted_rec)
        fold_weighted_f1.append(t_weighted_f1)

    # print average scores
    print()
    print("{} average results:".format(args.target))
    print("accuracy: {}".format(sum(fold_acc)/args.num_folds))
    print("weighted precision score: {}".format(sum(fold_weighted_prec) / args.num_folds))
    print("weighted recall score: {}".format(sum(fold_weighted_rec) / args.num_folds))
    print("weighted f1 score: {}".format(sum(fold_weighted_f1)/args.num_folds))

    # create output folder
    odir = './results/' + args.target + '/'
    if not os.path.exists(odir):
        os.makedirs(odir)

    # store per-fold scores
    with open(odir + args.target + '_BERT.csv', 'w') as out:
        out.write('acc,weighted_prec,weighted_rec,weighted_f1,epoch\n')
        for acc, weighted_prec, weighted_rec, weighted_f1, epoch in zip(fold_acc, fold_weighted_prec, fold_weighted_rec, fold_weighted_f1, fold_epoch):
            out.write(str(acc) + ',' + str(weighted_prec) + ',' + str(weighted_rec) + ',' + str(weighted_f1) + ',' + str(epoch) + '\n')


if __name__ == "__main__":
    main()
