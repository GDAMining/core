import argparse

from src.data import utils as data_utils


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='./path/to/raw/crawled/data/file', type=str, help='Target data to process.')
parser.add_argument('--odir', default='./path/to/proc/crawled/data/', type=str, help='Output directory.')

args = parser.parse_args()


def main():
    # process PubTator data
    data_utils.process_pubtator_data(args.data, args.odir)


if __name__ == "__main__":
    main()
