import argparse
import pandas as pd

from glob import glob
from tqdm import tqdm
from src.data import utils


parser = argparse.ArgumentParser()
parser.add_argument('--idir', default='./data/disgenet/raw/', type=str, help='Input directory.')
parser.add_argument('--odir', default='./data/disgenet/proc/', type=str, help='Output directory.')
parser.add_argument('--curated', default=True, type=bool, help='Whether to consider automatic or curated data.')


args = parser.parse_args()


def main():
    # gather DisGeNET paths
    data_paths = glob(args.idir + '*')

    # separate between automatic paths and curated ones
    automatic_paths = [path for path in data_paths if 'CURATED' not in path]
    curated_paths = [path for path in data_paths if 'CURATED' in path]

    if args.curated:  # consider curated data
        data_paths = curated_paths
        data_name = 'DISGENET_DATA_CURATED'
    else:  # consider automatic data
        data_paths = automatic_paths
        data_name = 'DISGENET_DATA_AUTOMATIC'

    # open DisGeNET data as DataFrame and concat
    print('Reading DisGeNET data ...')
    data = [pd.read_csv(path, sep='\t', header=0) for path in tqdm(data_paths, total=len(data_paths))]
    data = pd.concat(data)
    print('DisGeNET data read and concatenated')
    print('DisGeNET data has size {}'.format(data.shape[0]))

    # remove rows where AssociationType is 'Biomarker'
    print('Remove rows where AssociationType is Biomarker')
    data = data[data['Association_Type'] != 'Biomarker']
    print('DisGeNET data has now size {}'.format(data.shape[0]))

    # map AssociationType to EXPRESSION/OTHER
    preds = data['Association_Type'].unique().tolist()
    pred2bin = {pred: 'OTHER' for pred in preds}
    pred2bin['AlteredExpression'] = 'EXPRESSION'

    # convert AssociationType to EXPRESSION/OTHER
    data['Association_Type'] = data['Association_Type'].map(pred2bin)

    # store concatenated DisGeNET data
    print('Storing DisGeNET data ...')
    data.to_csv(args.idir + data_name + '.tsv', sep='\t', index=False)
    print('DisGeNET data stored!')

    # process DisGeNET data
    utils.process_disgenet(args.idir + data_name + '.tsv', args.odir)

    # change processed data attribute names
    print('Reading processed DisGeNET data ...')
    data = pd.read_csv(args.odir + data_name + '.csv', header=0)
    print('Processed DisGeNET data read!')

    # rename AssociationType to GCC
    print('Rename AssociationType attribute to GCC')
    data.rename(columns={'AssociationType': 'GCC'}, inplace=True)
    print('Attribute renamed!')

    # store renamed DisGeNET data
    print('Storing renamed DisGeNET data ...')
    data.to_csv(args.odir + data_name + '.csv', index=False)
    print('DisGeNET renamed data stored!')


if __name__ == "__main__":
    main()
