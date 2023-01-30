import os
import torch
import random
import argparse
import numpy as np

from tqdm import tqdm
from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification

from src.models import utils
from src.models.utils import InferenceDataset


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='./path/to/input/data/file', type=str, help='Target data to ingest.')
parser.add_argument('--bert', default='allenai/scibert_scivocab_uncased', type=str, help='BERT pre-trained model.')
parser.add_argument('--target', default='', type=str, help='Target factor to predict -- factors can be: "CGE", "CCS", "GCI".')
parser.add_argument('--gpu', default=0, type=int, help='Primary GPU.')
parser.add_argument('--seed', default=42, type=int, help='Random seed.')

args = parser.parse_args()


def main():

    # Randomization

    # set seed value(s) to reproduce results
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Directories Creation

    # create output dir
    odir = './data/inferred/' + args.target + '/'
    if not os.path.exists(odir):
        os.makedirs(odir)

    # GPU Setting

    # set GPU device
    if torch.cuda.is_available():  # use GPU -- set device to specified GPU number
        device = args.gpu
        print('GPU {} used: {}'.format(args.gpu, torch.cuda.get_device_name(args.gpu)))
    else:  # use CPU -- set device to -1
        device = -1
        print('GPUs not present, use CPU')

    # Data Processing

    # create dataset to perform inference
    dataset = utils.create_dataset(args.data)
    # restrict dataset to required data
    dataset = dataset[['PMID', 'PMIDYear', 'Sentence']]
    # get sentences for prediction
    sents = InferenceDataset(dataset['Sentence'].values.tolist())

    # BERT Setting

    # get model dir
    mdir = './trained/' + args.target + '_BERT'

    # set BERT tokenizer
    print('set BERT tokenizer')
    tokenizer = BertTokenizer.from_pretrained(args.bert)
    # set BERT for args.target
    print('set BERT for {} from {}'.format(args.target, mdir))
    model = BertForSequenceClassification.from_pretrained(mdir)
    # set pipeline for inference
    pipe = pipeline('text-classification', model=model, tokenizer=tokenizer, device=device)

    # Inference

    # set vars to store predicted labels w/ scores
    labels = []
    scores = []

    # perform inference over data
    print('perform inference over {} sentences'.format(len(dataset)))
    for pred in tqdm(pipe(sents, batch_size=16, max_length=512, truncation=True), total=len(dataset)):
        # store predictions
        labels += [pred['label']]
        scores += [pred['score']]
    print('inference process completed!')

    # add predictions to dataset
    dataset[args.target + '_label'] = labels
    dataset[args.target + '_score'] = scores

    print('store inferred data')
    # get dataset name
    dataset_name = args.data.split('/')[-1].split('.')[0]
    # store dataset
    dataset.to_csv(odir + dataset_name + '_' + args.target + '.csv', index=False)


if __name__ == "__main__":
    main()
