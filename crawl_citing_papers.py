import json
import argparse

from src.core import utils as core_utils
from src.data import utils as data_utils


parser = argparse.ArgumentParser()
parser.add_argument('--email', default='', type=str, help='Email address, required by NCBI.')
parser.add_argument('--data', default='./path/to/current/data/file', type=str, help='Target data to ingest.')
parser.add_argument('--outp', default='./path/to/citing/papers/file', help='Out path. JSON format required.')

args = parser.parse_args()


def main():
    # read input data and obtain (unique) PMIDs
    print('Reading {} ...'.format(args.data))
    data = core_utils.read_current_data(args.data)
    print('Data read!')
    pmids = data['PMID'].unique().tolist()
    # retrieve PMIDs that cite input PMIDs
    print('Retrieving PMIDs citing {} input PMIDs'.format(len(pmids)))
    citing_pmids = data_utils.retrieve_citing_pmids(pmids, args.email)
    print('Retrieved {} PMIDs!'.format(len(citing_pmids)))
    # fetch PubTator data for citing PMIDs
    data = data_utils.fetch_pubtator_data(pmids=citing_pmids, outformat='biocjson', concepts=['gene', 'disease'])
    # store data into outp
    print('Storing PubTator data associated w/ {} PMIDs ...'.format(len(data)))
    with open(args.outp, 'w') as out:
        json.dump(data, out, indent=4)
    print('PubTator data stored!')


if __name__ == "__main__":
    main()
