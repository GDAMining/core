import torch
import datetime
import pandas as pd

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset


# create vars to map GCC, CGE, CCS, and GCI values to indices
target2ix = {
    'GCC': {'OTHER': 0, 'EXPRESSION': 1},
    'CGE': {'NOTINF': 0, 'UP': 1, 'DOWN': 2},
    'CCS': {'NOTINF': 0, 'PROGRESSION': 1, 'REGRESSION': 2},
    'GCI': {'NOTINF': 0, 'OBSERVATION': 1, 'CAUSALITY': 2},
}


def mask_mentions(text, gene_start, gene_end, disease_start, disease_end):
    """
    Mask gene/disease mentions to avoid bias

    :param text: target text
    :param gene_start: gene start position
    :param gene_end: gene end position
    :param disease_start: disease start position
    :param disease_end: disease end position
    :return: text w/ gene/disease mentions replaced with placeholders
    """

    if gene_end < disease_start:  # gene mention before disease mention
        text = text[:gene_start] + '__gene__' + text[gene_end:disease_start] + '__disease__' + text[disease_end:]
    else:  # disease mention before gene mention
        text = text[:disease_start] + '__disease__' + text[disease_end:gene_start] + '__gene__' + text[gene_end:]
    return text


def create_dataset(data_path, target=None):
    """
    Prepare dataset to train/validate/test models

    :param data_path: input (annotated) data path
    :param target: target to consider
    :return: dataset
    """

    # read data from file
    data = pd.read_csv(data_path, header=0, keep_default_na=False)
    if target:  # curated data considered -- CGE, CCS, and GCI found within data
        if target != 'GCC':  # remove data where GCC attribute has OTHER as value
            data = data[data['GCC'] != 'OTHER']
        # restrict data to required information
        data = data[['Sentence', 'GeneStart', 'GeneEnd', 'DiseaseStart', 'DiseaseEnd', target]]
        # map target values to indices
        data[target] = data[target].map(target2ix[target])
    # mask gene/disease mentions within text
    data['Sentence'] = data.apply(lambda x: mask_mentions(x['Sentence'], x['GeneStart'], x['GeneEnd'], x['DiseaseStart'], x['DiseaseEnd']), axis=1)

    return data


class InferenceDataset(Dataset):
    """
    Convert data list to dataset to speed up inference
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def prepare_tokens(tokenizer, inputs):
    """
    Prepare dataset to train, validate, and test CEBERT

    :param tokenizer: BERT tokenizer
    :param inputs: train/valid/test data
    :return: encoded train/valid/test data
    """

    # encode text w/ BERT tokenizer
    etext = tokenizer.batch_encode_plus(
        inputs,
        max_length=512,
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',
        truncation=True
    )

    # return encoded text
    return etext['input_ids'], etext['attention_mask']


def prepare_dataset(tokenizer, inputs, targets, batch_size=32, train=True):
    """
    Prepare DataLoader functions to train, validate, and test CEBERT

    :param tokenizer: BERT tokenizer
    :param inputs: train/valid/test data
    :param targets: train/valid/test targets
    :param batch_size: batch size
    :param train: whether dataset used for training or validation/test
    :return: DataLoader for input data
    """

    # prepare tokens
    etext = prepare_tokens(tokenizer, inputs)
    # combine inputs into a TensorDataset -- input IDs, attention mask IDs, and targets
    dataset = TensorDataset(
        torch.tensor(etext[0]),
        torch.tensor(etext[1]),
        torch.tensor(targets.to_numpy())
    )

    if train:  # create DataLoader for training
        dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
    else:  # create DataLoader for validation/test
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

    return dataloader


def format_time(elapsed):
    """
    Take time in seconds and return string hh:mm:ss

    :param elapsed: elapsed time
    :return: formatted time
    """

    # round to the nearest second
    elapsed = int(round(elapsed))
    # format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed))
