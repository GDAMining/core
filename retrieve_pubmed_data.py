import argparse
import pandas as pd

from src.data import utils


parser = argparse.ArgumentParser()
parser.add_argument('--email', default='', type=str, help='Email address, required by NCBI.')
parser.add_argument('--data', default='./path/to/proc/crawled/data/file', type=str, help='Target data to expand w/ missing PubMed information.')

args = parser.parse_args()


def main():
    # read data
    print('Loading data...')
    data = pd.read_csv(args.data, header=0, keep_default_na=False)
    print('Data loaded!')

    # expand data w/ missing PubMed information
    print('Expanding data w/ (missing) PubMed information...')
    utils.fetch_pubmed_info(data, args.email)
    print('Data expanded w/ PubMed information!')

    # store expanded data
    print('Storing expanded data...')
    data.to_csv(args.data, index=False)
    print('Expanded data stored!')


if __name__ == "__main__":
    main()
