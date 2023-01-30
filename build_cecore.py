import os

from tqdm import tqdm
from src.core import utils
from src.core import cecore


def main():
    # create CE CORE dir
    odir = './db/'
    if not os.path.exists(odir):
        os.makedirs(odir)

    # get current data
    print('Load current data...')
    data = utils.read_current_data('./data/current/CECORE_DATA.csv')
    print('Current data loaded!')

    # prepare data for RDF processing
    print('Prepare data for RDF processing...')
    data4rdf = utils.prepare_data4rdf(data)
    print('Data prepared for RDF processing!')

    # init CECORE KG
    kg = cecore.CECORE()

    # iterate over prepared data and build CECORE RDF graph
    print('Convert data to RDF format...')
    for gene_cancer, gcs in tqdm(data4rdf.items(), total=len(data4rdf)):
        if gcs['hasType'] != 'UNRELIABLE':  # store data because RELIABLE
            kg.convert_gcs2rdf(gene_cancer, gcs)
    print('Data converted to RDF format!')

    # serialize CE CORE
    kg.serialize('./db/cecore.ttl')


if __name__ == "__main__":
    main()
