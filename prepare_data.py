import os

from glob import glob
from src.core import utils as core_utils


def main():
    # create dir to store current data version
    out_dir = './data/current/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # get manual data -- in the future, we might have multiple source of manual data and that explains the [] choice
    print('Loading manual data...')
    manual = [core_utils.read_manual_data('./data/manual/mdata.csv')]
    print('Manual data loaded!')

    # get crawled data -- crawled data comes from different sources and thus requires the use of []
    crawled_paths = glob('./data/crawled/*.csv')
    print('Loading crawled/inferred data...')
    crawled = [core_utils.read_crawled_data(path) for path in crawled_paths]
    print('Crawled/Inferred data loaded!')

    # combine manual and crawled data
    print('Combine manual and crawled data to obtain current data (version)')
    current = core_utils.combine_data(manual, crawled)

    # store current data (version)
    print('Storing current data...')
    current.to_csv(out_dir + 'data4kb.csv', index=False)
    print('Current data stored!')


if __name__ == "__main__":
    main()
