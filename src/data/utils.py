import json
import requests
import pandas as pd
import xml.etree.ElementTree as Etree

from tqdm import tqdm
from Bio import Entrez
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize


# set DisGeNET GDA attributes used to store data
gda_attrs = [
    'Gene', 'GeneID', 'GeneStart', 'GeneEnd',
    'Disease', 'DiseaseID', 'DiseaseStart', 'DiseaseEnd',
    'ScoreGDA', 'AssociationType', 'OriginalDB', 'Sentence', 'PMID', 'PMIDYear'
]

# set PubTator attributes used to store data
pub_attrs = [
    'GeneID', 'GeneStart', 'GeneEnd',
    'DiseaseID', 'DiseaseStart', 'DiseaseEnd',
    'Sentence', 'Curated', 'OriginalDB', 'PMID', 'PMIDYear', 'PMIDVenue', 'PMIDTitle', 'PMIDAbstract'
]


def get_span_tags(soup):  # TODO: check that adding tag.attrs['id'] to tags does not produce issues w/ DisGeNET data
    """
    Get "span" tags from processed sentence

    :param soup: BeautifulSoup processed
    :return: span tags divided into 'gene', 'disease', 'negexp', and 'variant'
    """

    # get "span" tags
    tags = {'gene': [], 'disease': [], 'negexp': [], 'variant': []}
    for tag in soup.find_all(name='span'):
        # get tag class
        tag_cls = tag.attrs['class']
        assert len(tag_cls) == 1  # check whether tag has multiple classes associated (shouldn't)
        # set tag text
        if tag_cls[0] == 'variant':  # variant tag has a different structure
            tag_txt = '<span class="' + tag_cls[0] + '" dbsnp="' + tag.attrs['dbsnp'] + '" genes_norm="' + tag.attrs['genes_norm'] + '">' + tag.text + '</span>'
            tag_id = tag.attrs['dbsnp']
        elif tag_cls[0] == 'negexp':  # negexp tag has a different structure
            tag_txt = '<span class="' + tag_cls[0] + '">' + tag.text + '</span>'
            tag_id = 'negex'
        else:  # gene/disease tags
            tag_txt = '<span class="' + tag_cls[0] + '" id="' + tag.attrs['id'] + '">' + tag.text + '</span>'
            tag_id = tag.attrs['id']
        # store (tag text, tag start position, tag end position, tag id)
        tags[tag_cls[0]].append([tag.text, tag.sourcepos, tag.sourcepos+len(tag_txt), tag_id])
    # return "span" tags
    return tags


def strip_variant_tags(sent, tags):
    """
    Strip "variant" tags from processed sentence

    :param sent: sentence associated with GDA
    :param tags: tags encoded within sentence
    :return: updated sentence and tags after stripping variant tag
    """

    # remove variant tags from sentences
    for i in range(len(tags['variant'])):
        # store sentence w/o variant tag
        sent = sent[:tags['variant'][i][1]] + tags['variant'][i][0] + sent[tags['variant'][i][2]:]

        # update variant tags positions
        for j in range(i+1, len(tags['variant'])):
            # update variant positions
            tags['variant'][j][1] = tags['variant'][j][1] - (tags['variant'][i][2] - tags['variant'][i][1]) + len(tags['variant'][i][0])
            tags['variant'][j][2] = tags['variant'][j][2] - (tags['variant'][i][2] - tags['variant'][i][1]) + len(tags['variant'][i][0])

        # update gene/disease tags positions
        if tags['variant'][i][1] < tags['gene'][0][1] and tags['variant'][i][1] < tags['disease'][0][1]:  # variant tag before than gene and disease
            # update gene positions
            tags['gene'][0][1] = tags['gene'][0][1] - (tags['variant'][i][2] - tags['variant'][i][1]) + len(tags['variant'][i][0])
            tags['gene'][0][2] = tags['gene'][0][2] - (tags['variant'][i][2] - tags['variant'][i][1]) + len(tags['variant'][i][0])
            # update disease positions
            tags['disease'][0][1] = tags['disease'][0][1] - (tags['variant'][i][2] - tags['variant'][i][1]) + len(tags['variant'][i][0])
            tags['disease'][0][2] = tags['disease'][0][2] - (tags['variant'][i][2] - tags['variant'][i][1]) + len(tags['variant'][i][0])

        if tags['variant'][i][1] > tags['gene'][0][1] and tags['variant'][i][1] < tags['disease'][0][1]:  # variant tag between gene and disease (gene first)
            # update disease positions
            tags['disease'][0][1] = tags['disease'][0][1] - (tags['variant'][i][2] - tags['variant'][i][1]) + len(tags['variant'][i][0])
            tags['disease'][0][2] = tags['disease'][0][2] - (tags['variant'][i][2] - tags['variant'][i][1]) + len(tags['variant'][i][0])

        if tags['variant'][i][1] < tags['gene'][0][1] and tags['variant'][i][1] > tags['disease'][0][1]:  # variant tag between gene and disease (disease first)
            # update gene positions
            tags['gene'][0][1] = tags['gene'][0][1] - (tags['variant'][i][2] - tags['variant'][i][1]) + len(tags['variant'][i][0])
            tags['gene'][0][2] = tags['gene'][0][2] - (tags['variant'][i][2] - tags['variant'][i][1]) + len(tags['variant'][i][0])
    # return updated sent and tags
    return sent, tags


def strip_gda_tags(sent, tags, ids=False):
    """
    Strip "gene"/"disease" tags from processed sentence

    :param sent: sentence associated with GDA
    :param tags: tags encoded within sentence
    :param ids: whether to return gene/disease IDs -- default False
    :return: updated sentence and gene/disease position (and IDs) after stripping gene/disease tags
    """

    # strip tags from sentence and store start/end positions of gene/disease mentions
    if tags['gene'][0][1] < tags['disease'][0][1]:  # gene entity comes first
        # store first part of sentence
        new_sent = sent[:tags['gene'][0][1]] + tags['gene'][0][0] + sent[tags['gene'][0][2]:tags['disease'][0][1]]
        # store gene/disease positions within new_sent
        gene_pos = [tags['gene'][0][1], tags['gene'][0][1]+len(tags['gene'][0][0])]
        disease_pos = [len(new_sent), len(new_sent)+len(tags['disease'][0][0])]
        # store second part of sentence
        new_sent = new_sent + tags['disease'][0][0] + sent[tags['disease'][0][2]:]
    else:  # disease entity comes first
        # store first part of sentence
        new_sent = sent[:tags['disease'][0][1]] + tags['disease'][0][0] + sent[tags['disease'][0][2]:tags['gene'][0][1]]
        # store gene/disease positions within new_sent
        disease_pos = [tags['disease'][0][1], tags['disease'][0][1]+len(tags['disease'][0][0])]
        gene_pos = [len(new_sent), len(new_sent)+len(tags['gene'][0][0])]
        # store second part of sentence
        new_sent = new_sent + tags['gene'][0][0] + sent[tags['gene'][0][2]:]

    if ids:  # return updated sentence and gene/disease positions and IDs
        return new_sent, gene_pos, disease_pos, tags['gene'][0][3], tags['disease'][0][3]
    else:  # return updated sentence and gene/disease positions
        return new_sent, gene_pos, disease_pos


def mstrip_gda_tags(sent, tags):
    """
    Strip multiple "gene"/"disease" tags from processed sentence

    :param sent: sentence associated with GDA
    :param tags: tags encoded within sentence
    :return: updated sentence and tags after stripping multiple gene/disease tags
    """

    # remove gene tags from sentences
    for i in range(len(tags['gene'])):
        # store sentence w/o gene tag
        sent = sent[:tags['gene'][i][1]] + tags['gene'][i][0] + sent[tags['gene'][i][2]:]

        # update subsequent gene tags positions
        for j in range(i+1, len(tags['gene'])):
            tags['gene'][j][1] = tags['gene'][j][1] - (tags['gene'][i][2] - tags['gene'][i][1]) + len(tags['gene'][i][0])
            tags['gene'][j][2] = tags['gene'][j][2] - (tags['gene'][i][2] - tags['gene'][i][1]) + len(tags['gene'][i][0])

        # update disease tags positions
        for j in range(len(tags['disease'])):
            if tags['gene'][i][1] < tags['disease'][j][1]:  # gene tag before than disease tag
                # update disease positions
                tags['disease'][j][1] = tags['disease'][j][1] - (tags['gene'][i][2] - tags['gene'][i][1]) + len(tags['gene'][i][0])
                tags['disease'][j][2] = tags['disease'][j][2] - (tags['gene'][i][2] - tags['gene'][i][1]) + len(tags['gene'][i][0])

        # update current gene tag end position
        tags['gene'][i][2] = tags['gene'][i][1]+len(tags['gene'][i][0])

    # remove disease tags from sentences
    for i in range(len(tags['disease'])):
        # store sentence w/o disease tag
        sent = sent[:tags['disease'][i][1]] + tags['disease'][i][0] + sent[tags['disease'][i][2]:]

        # update subsequent disease tags positions
        for j in range(i+1, len(tags['disease'])):
            tags['disease'][j][1] = tags['disease'][j][1] - (tags['disease'][i][2] - tags['disease'][i][1]) + len(tags['disease'][i][0])
            tags['disease'][j][2] = tags['disease'][j][2] - (tags['disease'][i][2] - tags['disease'][i][1]) + len(tags['disease'][i][0])

        # update gene tags positions
        for j in range(len(tags['gene'])):
            if tags['disease'][i][1] < tags['gene'][j][1]:  # disease tag before than gene tag
                # update gene positions
                tags['gene'][j][1] = tags['gene'][j][1] - (tags['disease'][i][2] - tags['disease'][i][1]) + len(tags['disease'][i][0])
                tags['gene'][j][2] = tags['gene'][j][2] - (tags['disease'][i][2] - tags['disease'][i][1]) + len(tags['disease'][i][0])

        # update current disease tag end position
        tags['disease'][i][2] = tags['disease'][i][1]+len(tags['disease'][i][0])
    # return updated sent and tags
    return sent, tags


def keep_most_recent(data):  # TODO: keys var does not contain 'AssociationType' -- refactor this when working w/ several association types
    """
    Remove duplicates in terms of PMID Year -- keep most recent instances

    :param data: pandas DataFrame containing GDA information
    :return: pandas DataFrame w/o duplicates
    """

    # sort instances by PMID Year (desc)
    proc_data = data.sort_values(by=['PMIDYear'], ascending=False)
    # fix the set of attributes used to remove duplicates
    keys = ['GeneID', 'GeneStart', 'GeneEnd', 'DiseaseID', 'DiseaseStart', 'DiseaseEnd', 'Sentence']
    # return data w/o duplicates
    return proc_data.drop_duplicates(subset=keys, keep='first')


def remove_overlaps(data):
    """
    Remove instances with gene and disease mentions overlapping within text

    :param data: pandas DataFrame containing GDA information
    :return: pandas DataFrame w/o overlapping instances
    """

    ixs = []
    # iterate over rows and check for overlaps between gene and disease mentions
    for ix, gda in data.iterrows():
        if set(range(gda['GeneStart'], gda['GeneEnd'])).intersection(range(gda['DiseaseStart'], gda['DiseaseEnd'])):  # mentions overlap -- remove
            ixs += [ix]
    # drop rows with overlaps between gene and disease mentions
    out_data = data.drop(ixs)
    # return data w/o overlaps
    return out_data


def process_disgenet(data_path, out_dir):
    """
    Process DisGeNET data and make it compliant w/ CECORE format

    :param data_path: path to target data
    :param out_dir: path to output directory
    :return: None
    """

    print('Read DisGeNET data')
    # read DisGeNET data as pandas DataFrame and store data fname
    gda_data = pd.read_csv(data_path, sep='\t', header=0)
    data_name = data_path.split('/')[-1].split('.')[0]

    print('Start processing...')
    # set processed GDAs var
    proc_data = []
    print('Process DisGeNET data and extract GDAs')
    # iterate over rows and process Sentence w/ BeautifulSoup
    for ix, gda in tqdm(gda_data.iterrows()):
        # check if Sentence is not empty
        if type(gda['Sentence']) == str:
            # process the sentence w/ BeautifulSoup
            soup = BeautifulSoup(gda['Sentence'], features='html.parser')
            # get "span" tags
            tags = get_span_tags(soup)
            # check if negexp tag has size != 0
            if len(tags['negexp']) != 0:  # skip sentence that contains negative expressions
                continue
            # check if gene/disease tags have size == 0
            if len(tags['gene']) == 0 or len(tags['disease']) == 0:
                continue
            # check if gene/disease tags have size > 1
            if len(tags['gene']) > 1 or len(tags['disease']) > 1:  # strip multiple tags and keep first one (design choice)
                sent = gda['Sentence']
                # strip tags from sentence and store start/end positions of (multiple) gene/disease mentions
                new_sent, tags = mstrip_gda_tags(sent, tags)
                gene_pos = tags['gene'][0][1:3]  # TODO: can we do better than this?
                disease_pos = tags['disease'][0][1:3]  # TODO: same as above
            else:  # keep sentence
                sent = gda['Sentence']
                if tags['variant']:  # variant tag found within sentence
                    # remove variant tags from sentences and update sent and tags accordingly
                    sent, tags = strip_variant_tags(sent, tags)
                # strip tags from sentence and store start/end positions of gene/disease mentions
                new_sent, gene_pos, disease_pos = strip_gda_tags(sent, tags)
            # store data after processing
            proc_gda = gda.tolist()
            proc_data.append(proc_gda[0:2] + gene_pos + proc_gda[2:4] + disease_pos + proc_gda[4:7] + [new_sent] + proc_gda[8:])
    print('Process finished!')

    # convert proc_data to DataFrame
    proc_data = pd.DataFrame(proc_data, columns=gda_attrs)
    # remove duplicates by keeping most recent instances in terms of PMID Year
    out_data = keep_most_recent(proc_data)
    print('Processing finished!')

    print('Store processed data')
    # store data
    out_data.to_csv(out_dir+ '/' + data_name + '.csv', index=False)

    # compute data stats
    print('Compute data statistics')
    # count the number of instances per association type
    out_count = out_data.groupby('AssociationType')['AssociationType'].count().reset_index(name='count')
    print('Number of (processed) instances per association type')
    print(out_count.to_markdown())


def fetch_pubmed_info(data, email):
    """
    Expand data w/ PubMed missing information -- title, abstract, venue, year

    :param data: data to expand
    :param email: email address required by NCBI
    :return: None
    """

    # set email address -- required by NCBI
    Entrez.email = email
    # set batch size -- required to perform URL lookup requests
    batch_size = 10000

    # expand data structure w/ required attributes
    if 'PMIDYear' not in data.columns:  # add PMIDYear column to data
        data['PMIDYear'] = 0
    if 'PMIDVenue' not in data.columns:  # add PMIDVenue column to data
        data['PMIDVenue'] = ''
    if 'PMIDTitle' not in data.columns:  # add PMIDTitle column to data
        data['PMIDTitle'] = ''
    if 'PMIDAbstract' not in data.columns:  # add PMIDAbstract column to data
        data['PMIDAbstract'] = ''

    # get (unique) PMIDs from data
    pmids = data['PMID'].unique().tolist()
    # compute number of PMIDs
    count = len(pmids)

    # post NCBI query to store PMIDs
    search_handle = Entrez.epost(db='pubmed', id=','.join(map(str, pmids)))
    search_results = Entrez.read(search_handle)
    webenv, query_key = search_results['WebEnv'], search_results['QueryKey']

    # fetch results in batches of batch_size entries at once
    for start in range(0, count, batch_size):
        # compute end var
        end = min(count, start+batch_size)
        print('Downloading records from {} to {} out of {} PMIDs'.format(start+1, end, count))
        # perform single URL lookup -- fetch multiple PubMed articles based on PMIDs, which are returned as a handle
        handle = Entrez.efetch(db='pubmed', retmode='text', rettype='xml', retstart=start, retmax=batch_size, webenv=webenv, query_key=query_key)
        records = Entrez.read(handle)['PubmedArticle']  # restrict to PubmedArticle section -- contains the required PubMed information

        # iterate over records and expand data w/ PubMed information
        for paper in tqdm(records, total=len(records)):
            # get PubMed information from paper
            pmid = int(paper['MedlineCitation']['PMID'])
            year = int([pub_data['Year'] for pub_data in paper['PubmedData']['History'] if pub_data.attributes['PubStatus'] == 'pubmed'][0])
            venue = paper['MedlineCitation']['Article']['Journal']['ISOAbbreviation']
            if 'ArticleTitle' in paper['MedlineCitation']['Article']:  # ArticleTitle found within paper
                title = paper['MedlineCitation']['Article']['ArticleTitle']
            else:  # ArticleTitle not found within paper
                title = ''
            if 'Abstract' in paper['MedlineCitation']['Article']:  # Abstract found within paper
                abstract = ' '.join(paper['MedlineCitation']['Article']['Abstract']['AbstractText'])
            else:  # Abstract not found within paper
                abstract = ''

            # update data corresponding to current PMID w/ PubMed information
            data.loc[data['PMID'] == pmid, ['PMIDYear', 'PMIDVenue', 'PMIDTitle', 'PMIDAbstract']] = year, venue, title, abstract


def retrieve_citing_pmids(pmids, email):
    """
    Retrieve PMIDs of papers citing input PMIDs within PubMed

    :param pmids: input PMIDs
    :param email: email address required by NCBI
    :return: PMIDs of papers that cite input PMIDs
    """

    # set email address -- required by NCBI
    Entrez.email = email
    # set batch size, required to perform URL lookup requests -- CAVEAT: greater than 1000 produces runtime error w/ ELink
    batch_size = 1000

    # compute number of PMIDs
    count = len(pmids)

    cpmids = []
    # fetch results in batches of batch_size entries at once
    for start in range(0, count, batch_size):
        # compute end var
        end = min(count, start + batch_size)
        print('Retrieving PMIDs citing input PMIDs from {} to {}'.format(start + 1, end, count))

        # post NCBI query to store PMIDs
        search_handle = Entrez.epost(db='pubmed', id=','.join(map(str, pmids[start:end])))
        search_results = Entrez.read(search_handle)
        webenv, query_key = search_results['WebEnv'], search_results['QueryKey']

        # perform single URL lookup -- map input PMIDs to citing PMIDs, which are returned as a handle
        handle = Entrez.elink(db='pubmed', dbfrom='pubmed', linkname='pubmed_pubmed_citedin', retmode='text', rettype='xml', webenv=webenv, query_key=query_key)
        records = Entrez.read(handle)[0]['LinkSetDb'][0]['Link']  # restrict to Link section -- contains the citing PMIDs

        # iterate over records and store citing PMIDs
        cpmids += [paper['Id'] for paper in tqdm(records, total=len(records))]

    # remove duplicates
    cpmids = list(set(cpmids))
    print('Retrieved {} PMIDs of papers that cite input PMIDs'.format(len(cpmids)))
    # return citing PMIDs
    return cpmids


def fetch_pubtator_data(pmids, outformat, concepts):
    """
    Retrieve PubTator annotations of PMIDs

    :param pmids: PMIDs to obtain annotations for
    :param outformat: output format
    :param concepts: annotation concept types
    :return: PubTator annotations of PMIDs
    """

    # set mapper from MeSH IDs to UMLS CUIs -- restrict to 'neoplastic process' UMLS semantic type (i.e., T191)
    print('Building mapper from MeSH and OMIM IDs to UMLS CUIs...')
    mapper = build_disease_mapper('./src/data/mappings/mrconso-file', './src/data/mappings/mrsty-file', ['T191'])
    print('Mapper from MeSH and OMIM IDs to UMLS CUIs built!')

    # set batch size -- required to perform URL lookup requests
    batch_size = 1000  # max size for POST request
    # compute number of PMIDs
    count = len(pmids)

    # prepare data var to store processed data w/ PubTator
    data = {}

    # fetch results in batches of batch_size entries at once
    for start in range(0, count, batch_size):
        # compute end var
        end = min(count, start + batch_size)
        print('Downloading records from {} to {} out of {} PMIDs'.format(start + 1, end, count))

        # set parameters used to perform URL lookup requests
        params = {'pmids': pmids[start:end]}

        # perform POST request
        req = requests.post('https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/'+outformat, json=params)
        if req.status_code != 200:  # HTTP Exception
            print('[Error]: HTTP code ' + str(req.status_code))
            raise Exception
        else:  # HTTP POST request completed -- retrieved annotated publications by PubTator
            annot_pubs = [json.loads(annot_pub) for annot_pub in req.text.splitlines()]
        # iterate over annotated publications
        for pub in tqdm(annot_pubs, total=len(annot_pubs)):
            # store publication data
            data[pub['pmid']] = {'PMIDVenue': pub['journal'], 'PMIDYear': pub['year'], 'PMIDTitle': '', 'PMIDAbstract': '', 'annotations': []}

            # store annotated data and tag gene/disease mentions within text as <span class=""disease/gene"" id=""disease-/gene-id"">mention</span>
            for psg in pub['passages']:
                # set passage data
                section = psg['infons']['section']
                text = psg['text']
                poffset = psg['offset']

                # store section data -- raw text w/o PubTator annotations
                data[pub['pmid']]['PMID'+section] = text

                # prepare vars to store passage annotations
                mids, mtypes = [], []
                mstarts, mends = [], []
                # iterate over passage annotations and store annotated data
                for annot in psg['annotations']:
                    # store mention ID -- either MESH or Entrez ID
                    mids.append(annot['infons']['identifier'])
                    # store mention type -- either Disease or Gene
                    mtypes.append(annot['infons']['type'].lower())

                    # sanity check
                    assert len(annot['locations']) == 1
                    # store mention offset and length
                    moffset = annot['locations'][0]['offset']
                    mlength = annot['locations'][0]['length']

                    # store mention start position -- passage offset must be subtracted
                    mstarts.append(moffset-poffset)
                    # store mention end position -- passage offset must be subtracted
                    mends.append(moffset+mlength-poffset)

                # set vars required to encode text w/ gene/disease tags
                etext= ''
                eoffset = 0
                # encode text w/ gene/disease tags
                for ix in range(len(mids)):
                    if mtypes[ix] not in concepts:  # mention refers to a different concept -- skip it
                        continue
                    if not mids[ix]:  # mention ID not found by PubTator -- skip it
                        continue

                    if mtypes[ix] == 'disease':  # mention type belongs to 'disease'
                        # get MeSH or OMIM ID from mention ID
                        mid = mids[ix].split(':')[1]  # mention ID stored as resource:id -- strip resource: and keep id
                        if mid in mapper:  # mention ID refers to cancer disease -- map to UMLS CUI
                            cui = mapper[mid]
                            etext += text[eoffset:mstarts[ix]] + '<span class="' + mtypes[ix] + '" id="' + cui + '">'
                        else:  # mention ID does not refer to cancer disease -- skip mention
                            continue
                    else:  # mention type belongs to 'gene'
                        etext += text[eoffset:mstarts[ix]]+'<span class="'+mtypes[ix]+'" id="'+mids[ix]+'">'
                    etext += text[mstarts[ix]:mends[ix]]+'</span>'
                    # update encoding offset position w/ mends[ix]
                    eoffset = mends[ix]
                # store PubTator data
                data[pub['pmid']]['annotations'] += [etext]
    # return data
    return data


def build_disease_mapper(mrconsofp, mrstyfp, semtypes=None):
    """
    Build mapper from MeSH and OMIM IDs to the corresponding UMLS CUIs -- restrict to input semantic types

    :param mrconsofp: path to MRCONSO.RRF
    :param mrstyfp: path to MRSTY.RRF
    :param semtypes: (list of) UMLS semantic types used to restrict mapper scope -- when None all semantic types are considered
    :return: dict mapping MeSH and OMIM IDs to UMLS CUIs
    """

    # read MRCONSO.RRF
    with open(mrconsofp, 'r') as f:
        mrconso = f.readlines()
    # process MRCONSO to keep entries associated w/ MeSH and OMIM and using ENG
    mrconso = [row.split('|') for row in mrconso]
    mrconso = {rdata[10]: rdata[0] for rdata in mrconso if rdata[1] == 'ENG' and rdata[11] in ['MSH', 'OMIM']}

    # read MRSTY.RRF
    with open(mrstyfp, 'r') as f:
        mrsty = f.readlines()
    # process MRSTY to keep entries associated w/ input semantic types
    mrsty = [row.split('|') for row in mrsty]
    mrsty = {rdata[0]: rdata[1] for rdata in mrsty if rdata[1] in semtypes}

    # build mapper from MeSH and OMIM IDs to UMLS IDs
    mapper = {did: cui for did, cui in mrconso.items() if cui in mrsty}
    # return mapper
    return mapper


def process_pubtator_data(data_path, out_dir):
    """
    Process PubTator data and make it compliant w/ CECORE format

    :param data_path: path to target data
    :param out_dir: path to output directory
    :return: None
    """

    print('Reading {} ...'.format(data_path))
    # read PubTator data as JSON dict and store data fname
    with open(data_path, 'r') as dataf:
        data = json.load(dataf)
    data_name = data_path.split('/')[-1].split('.')[0]
    print('Data read!')

    print('Start processing...')
    # set processed data var
    proc_data = []
    print('Process PubTator data and extract annotations')
    # iterate over PMIDs and process annotations w/ BeautifulSoup
    for pmid in tqdm(data.keys(), total=len(data)):
        for annot in data[pmid]['annotations']:
            # check if annotated sentence is not empty
            if annot:
                if not data[pmid]['PMIDYear']:  # convert None to 0 -- keep consistent data types
                    data[pmid]['PMIDYear'] = 0
                # tokenize annotation into sentences
                for sent in sent_tokenize(annot):
                    # process annotated sentence w/ BeautifulSoup
                    soup = BeautifulSoup(sent, features='html.parser')
                    # get "span" tags
                    tags = get_span_tags(soup)
                    # check if gene/disease tags have size == 0
                    if len(tags['gene']) == 0 or len(tags['disease']) == 0:
                        continue
                    # check if gene/disease tags have size > 1
                    if len(tags['gene']) > 1 or len(tags['disease']) > 1:  # strip multiple tags and combine each gene tag w/ each disease tag
                        # strip tags from sentence and store start/end positions of (multiple) gene/disease mentions
                        new_sent, tags = mstrip_gda_tags(sent, tags)
                        for gene_tag in tags['gene']:  # iterate over genes
                            for disease_tag in tags['disease']:  # iterate over diseases
                                # get gene information
                                gene_pos, gene_id = gene_tag[1:3], gene_tag[3]
                                # strip multiple gene IDs and keep first -- TODO: is this a good idea?
                                gene_id = gene_id.split(';')[0]
                                # get disease information
                                disease_pos, disease_id = disease_tag[1:3], disease_tag[3]

                                # store data after processing
                                proc_data += [[
                                    gene_id, gene_pos[0], gene_pos[1],
                                    disease_id, disease_pos[0], disease_pos[1],
                                    new_sent, False, 'PUBMED', pmid, data[pmid]['PMIDYear'], data[pmid]['PMIDVenue'], data[pmid]['PMIDTitle'], data[pmid]['PMIDAbstract']
                                ]]
                    else:  # gene/disease tags have size == 1 -- keep as is
                        # strip tags from sentence and store start/end positions of gene/disease mentions
                        new_sent, gene_pos, disease_pos, gene_id, disease_id = strip_gda_tags(sent, tags, ids=True)
                        # strip multiple gene IDs and keep first -- TODO: is this a good idea?
                        gene_id = gene_id.split(';')[0]

                        # store data after processing
                        proc_data += [[
                            gene_id, gene_pos[0], gene_pos[1],
                            disease_id, disease_pos[0], disease_pos[1],
                            new_sent, False, 'PUBMED', pmid, data[pmid]['PMIDYear'], data[pmid]['PMIDVenue'], data[pmid]['PMIDTitle'], data[pmid]['PMIDAbstract']
                        ]]
    print('PubTator process finished!')

    # convert proc to DataFrame
    proc_data = pd.DataFrame(proc_data, columns=pub_attrs)
    # remove duplicates by keeping most recent instances in terms of PMID Year
    out_data = keep_most_recent(proc_data)
    # remove instances where gene and disease mentions overlap within text (if any)
    out_data = remove_overlaps(out_data)
    print('Processing finished!')

    print('Store processed data')
    # store data
    out_data.to_csv(out_dir + '/' + data_name + '.csv', index=False)

    # count the number of instances per association type
    print('Number of (processed) PubTator sentences: {}'.format(out_data.shape[0]))
