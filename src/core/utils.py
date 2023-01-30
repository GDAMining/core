import pandas as pd

from tqdm import tqdm


# set mapping between DisGeNET sources and DisGeNET
src2db = {
    'BEFREE': 'https://www.disgenet.org/',
    'LHGDN': 'https://www.disgenet.org/'
}


# set CECORE attributes required to prepare data for ingestion
core_attrs = [
    'GeneID', 'GeneStart', 'GeneEnd',
    'DiseaseID', 'DiseaseStart', 'DiseaseEnd',
    'Sentence', 'Curated', 'OriginalDB', 'PMID', 'PMIDYear', 'PMIDVenue', 'PMIDTitle', 'PMIDAbstract',
    'CGE_label', 'CGE_score', 'CCS_label', 'CCS_score', 'GCI_label', 'GCI_score', 'GCC_label', 'GCC_score'
]


# set threshold scores required to tag facts
thr = {
    'NOTINF': 0.7,
    'EGR': 0.4, 
    'AGT': 0.4,
}


def read_manual_data(data_path):
    """
    Read manually curated data

    :param data_path: path to data set
    :return: pandas DataFrame containing manual data
    """

    data = pd.read_csv(data_path, header=0, keep_default_na=False)
    # rename CGE, CCS, GCI, and GCC and assign the associated scores (set to 1.0)
    data.rename(columns={'CGE': 'CGE_label', 'CCS': 'CCS_label', 'PT': 'GCI_label', 'CE': 'GCC_label'}, inplace=True)
    data['CGE_score'] = 1.0
    data['CCS_score'] = 1.0
    data['GCI_score'] = 1.0
    data['GCC_score'] = 1.0
    # replace 'UNKNOWN' w/ 'NOTINF' -- if any
    data = data.replace(to_replace='UNKNOWN', value='NOTINF')
    # assign curated attribute to data and set to True
    data['Curated'] = True
    # return data
    return data


def read_crawled_data(data_path):
    """
    Read automatically extracted data

    :param data_path: path to data set
    :return: pandas DataFrame containing crawled data
    """

    data = pd.read_csv(data_path, header=0)
    # get data name
    data_name = data_path.split('/')[-1].split('.')[0]
    # get inferred data associated with crawled data
    data_cge = pd.read_csv('../cecore/data/inferred/CGE/' + data_name + '_CGE.csv', header=0, keep_default_na=False)
    data_ccs = pd.read_csv('../cecore/data/inferred/CCS/' + data_name + '_CCS.csv', header=0, keep_default_na=False)
    data_pt = pd.read_csv('../cecore/data/inferred/PT/' + data_name + '_PT.csv', header=0, keep_default_na=False)
    data_ce = pd.read_csv('../cecore/data/inferred/CE/' + data_name + '_CE.csv', header=0, keep_default_na=False)
    # concatenate inferred data with crawled data
    data['CGE_label'] = data_cge['CGE_label']
    data['CGE_score'] = data_cge['CGE_score']
    data['CCS_label'] = data_ccs['CCS_label']
    data['CCS_score'] = data_ccs['CCS_score']
    data['GCI_label'] = data_pt['PT_label']
    data['GCI_score'] = data_pt['PT_score']
    data['GCC_label'] = data_ce['CE_label']
    data['GCC_score'] = data_ce['CE_score']
    # replace 'UNKNOWN' w/ 'NOTINF' -- if any
    data = data.replace(to_replace='UNKNOWN', value='NOTINF')
    # assign curated attribute to data and set to False
    data['Curated'] = False
    # return data
    return data


def combine_data(manual, crawled):
    """
    Combine manual and crawled data

    :param manual:  manually annotated data ([DataFrame, ...])
    :param crawled: automatically extracted data ([DataFrame, ...])
    :return: a combined DataFrame that stores both manual and crawled data together
    """

    # concatenate manual data
    mdata = pd.concat(manual)
    # concatenate crawled data
    cdata = pd.concat(crawled)

    # set subset of attributes used to compare manual and automatic data -- CAVEAT: GeneID and DiseaseID describe the proper GCS
    subset = ['GeneID', 'GeneStart', 'GeneEnd', 'DiseaseID', 'DiseaseStart', 'DiseaseEnd', 'Sentence', 'PMID']
    # drop automatic data when equal to manual data -- use subset to restrict to appropriate attributes
    cdata = pd.merge(cdata, mdata[subset], how='left', on=subset, indicator=True)
    cdata = cdata[cdata['_merge'] == 'left_only']
    cdata = cdata.drop(columns=['_merge'])

    # concatenate manual and crawled data
    data = pd.concat([mdata, cdata])
    # keep data where CGE != NOTINF and GCC != OTHER
    data = data[(data['CGE_label'] != 'NA') & (data['GCC_label'] != 'OTHER')]
    #  restrict data to CECORE attributes required to prepare data for ingestion
    return data[core_attrs]


def read_current_data(data_path):
    """
    Read current data (manual+crawled)

    :param data_path: path to data set
    :return: pandas DataFrame containing manual data
    """

    data = pd.read_csv(data_path, header=0)
    # return data
    return data


def prepare_data4rdf(current): # TODO: fix this one after you fixed stuff below
    """
    Prepare manual and crawled data for RDF processing

    :param current: current data (DataFrame) -- manually annotated + automatically extracted
    :return: a dict where each GCS (Gene Cancer Status) contains GCS- and sentence-level data to be used for RDF graph construction
    """

    # convert data from DataFrame to dict record-wise
    data = current.to_dict(orient='records')

    data4rdf = {}
    # iterate over data and prepare it for RDF ingestion
    for item in tqdm(data, total=len(data)):
        # get GeneID and DiseaseID and check whether GCS is stored in data4rdf
        gcs = (item['GeneID'], item['DiseaseID'])
        if gcs not in data4rdf:  # GCS not found within data4rdf -- create params
            data4rdf[gcs] = init_gcs()
        # update GCS w/ current item
        update_gcs(data4rdf[gcs], item)
    # iterate over GCSs and set GCS type
    for data_gcs in tqdm(data4rdf.values(), total=len(data4rdf)):
        tag_gcs2info(data_gcs)  # check GCS sufficiency
        if data_gcs['isInformative']:
            tag_gcs2type(data_gcs)  # check GCS consistency

    # return data prepared for RDF processing
    return data4rdf


def init_gcs():  # TODO: needs to be updated?
    """
    Initialize GCS and store params

    :return: Initialized GCS
    """

    # init GCS
    gcs = {
        'CGE': {'Support': {'UP': 0, 'DOWN': 0}, 'Score': {'UP': 0.0, 'DOWN': 0.0}, 'Likelihood': {'UP': 0.0, 'DOWN': 0.0}},
        'CCS': {'Support': {'PROGRESSION': 0, 'REGRESSION': 0, 'NOTINF': 0}, 'Score': {'PROGRESSION': 0.0, 'REGRESSION': 0.0, 'NOTINF': 0.0}, 'Likelihood': {'PROGRESSION': 0.0, 'REGRESSION': 0.0, 'NOTINF': 0.0}},
        'GCI': {'Support': {'OBSERVATION': 0, 'CAUSALITY': 0, 'NOTINF': 0}, 'Score': {'OBSERVATION': 0.0, 'CAUSALITY': 0.0, 'NOTINF': 0.0}, 'Likelihood': {'OBSERVATION': 0.0, 'CAUSALITY': 0.0, 'NOTINF': 0.0}},
        'EGR': {'Support': {'PASSIVE': 0, 'ACTIVE': 0}, 'Score': {'PASSIVE': 0.0, 'ACTIVE': 0.0}, 'Likelihood': {'PASSIVE': 0.0, 'ACTIVE': 0.0}},
        'AGT': {'Support': {'ONCOGENE': 0, 'TSG': 0}, 'Score': {'ONCOGENE': 0.0, 'TSG': 0.0}, 'Likelihood': {'ONCOGENE': 0.0, 'TSG': 0.0}},
        'hasType': 'UNRELIABLE', 'isInformative': False,
        'sents': []
    }
    # return GCS
    return gcs


def update_gcs(data_gcs, item):
    """
    Update GCS w/ current item

    :param data_gcs: current GCS data
    :param item: target sentence to ingest
    :return: None
    """

    # get annotations
    cge = item['CGE_label']
    cge_score = item['CGE_score']
    ccs = item['CCS_label']
    ccs_score = item['CCS_score']
    gci = item['GCI_label']
    gci_score = item['GCI_score']

    # update CGE, CCS, and GCI data at GCS-level
    data_gcs['CGE']['Support'][cge] += 1
    data_gcs['CGE']['Score'][cge] += cge_score
    data_gcs['CGE']['Likelihood']['UP'] = compute_likelihood(data_gcs['CGE']['Score'], 'UP')
    data_gcs['CGE']['Likelihood']['DOWN'] = compute_likelihood(data_gcs['CGE']['Score'], 'DOWN')

    data_gcs['CCS']['Support'][ccs] += 1
    data_gcs['CCS']['Score'][ccs] += ccs_score
    data_gcs['CCS']['Likelihood']['PROGRESSION'] = compute_likelihood(data_gcs['CCS']['Score'], 'PROGRESSION')
    data_gcs['CCS']['Likelihood']['REGRESSION'] = compute_likelihood(data_gcs['CCS']['Score'], 'REGRESSION')
    data_gcs['CCS']['Likelihood']['NOTINF'] = compute_likelihood(data_gcs['CCS']['Score'], 'NOTINF')

    data_gcs['GCI']['Support'][gci] += 1
    data_gcs['GCI']['Score'][gci] += gci_score
    data_gcs['GCI']['Likelihood']['OBSERVATION'] = compute_likelihood(data_gcs['GCI']['Score'], 'OBSERVATION')
    data_gcs['GCI']['Likelihood']['CAUSALITY'] = compute_likelihood(data_gcs['GCI']['Score'], 'CAUSALITY')
    data_gcs['GCI']['Likelihood']['NOTINF'] = compute_likelihood(data_gcs['GCI']['Score'], 'NOTINF')

    if gci != 'NOTINF': #and ccs != 'NOTINF':  # update EGR
        if gci == 'OBSERVATION':  # sentence supports a passive role for target gene
            data_gcs['EGR']['Support']['PASSIVE'] += 1
            data_gcs['EGR']['Score']['PASSIVE'] += gci_score
        else:  # sentence supports an active role for target gene
            data_gcs['EGR']['Support']['ACTIVE'] += 1
            data_gcs['EGR']['Score']['ACTIVE'] += gci_score
        data_gcs['EGR']['Likelihood']['PASSIVE'] = compute_likelihood(data_gcs['EGR']['Score'], 'PASSIVE')
        data_gcs['EGR']['Likelihood']['ACTIVE'] = compute_likelihood(data_gcs['EGR']['Score'], 'ACTIVE')

    if gci == 'CAUSALITY' and ccs != 'NOTINF':  # update AGT
        if (cge == 'UP' and ccs == 'PROGRESSION') or (cge == 'DOWN' and ccs == 'REGRESSION'):  # ONCOGENE
            data_gcs['AGT']['Support']['ONCOGENE'] += 1
            data_gcs['AGT']['Score']['ONCOGENE'] += cge_score * ccs_score
        if (cge == 'DOWN' and ccs == 'PROGRESSION') or (cge == 'UP' and ccs == 'REGRESSION'):  # TSG
            data_gcs['AGT']['Support']['TSG'] += 1
            data_gcs['AGT']['Score']['TSG'] += cge_score * ccs_score
        data_gcs['AGT']['Likelihood']['ONCOGENE'] = compute_likelihood(data_gcs['AGT']['Score'], 'ONCOGENE')
        data_gcs['AGT']['Likelihood']['TSG'] = compute_likelihood(data_gcs['AGT']['Score'], 'TSG')

    # store sentence-level data
    data_gcs['sents'] += [prepare_sent_data(item)]


def compute_likelihood(task_score, target):
    """
    Compute the likelihood of having such target associated w/ current GCS -- weighted by prediction scores

    :param task_score: the cumulative score associated w/ task (i.e., CGE, CCS, or GCI score)
    :param target: target value
    :return: updated likelihood based on target value and task cumulative score
    """

    # compute target score
    prob = task_score[target] / sum(task_score.values())
    return prob


def tag_gcs2info(data_gcs):
    """
    Tag GCS as either 'informative' or 'not-informative' based on threshold function -- threshold function works as factor3_prob > thr

    :param data_gcs: current GCS data
    :return: None
    """

    # CCS threshold
    if data_gcs['CCS']['Likelihood']['NOTINF'] > thr['NOTINF']:  # CCS above threshold -- set trust to False (GCS == 'not-informative')
        trust = False
    # GCI threshold
    elif data_gcs['GCI']['Likelihood']['NOTINF'] > thr['NOTINF']:  # GCI above threshold -- set trust to False (GCS == 'not-informative')
        trust = False
    else:  # CCS and GCI below threshold -- set trust to True (GCS == 'informative')
        trust = True

    # GCS tagged as trust var
    data_gcs['isInformative'] = trust


def tag_gcs2type(data_gcs):
    """
    Tag GCS type as either 'BIOMARKER', 'ONCOGENE', 'TSG', or 'UNRELIABLE' based on threshold function(s)

    :param data_gcs: current GCS data
    :return: None
    """

    if abs(data_gcs['EGR']['Likelihood']['ACTIVE'] - data_gcs['EGR']['Likelihood']['PASSIVE']) <= thr['EGR']:  # EGR below/equal threshold -- set GCS type to 'UNRELIABLE'
        gcs_type = 'UNRELIABLE'
    else:  # EGR above threshold
        if data_gcs['EGR']['Likelihood']['PASSIVE'] > data_gcs['EGR']['Likelihood']['ACTIVE']:  # set GCS type to 'BIOMARKER'
            gcs_type = 'BIOMARKER'
        else:  # check GCS type
            if abs(data_gcs['AGT']['Likelihood']['ONCOGENE'] - data_gcs['AGT']['Likelihood']['TSG']) <= thr['AGT']:  # set GCS type to 'UNRELIABLE'
                gcs_type = 'UNRELIABLE'
            else:  # set GCS type as ONCOGENE or TSG
                if data_gcs['AGT']['Likelihood']['ONCOGENE'] > data_gcs['AGT']['Likelihood']['TSG']:  # GCS has ONCOGENE type
                    gcs_type = 'ONCOGENE'
                else:  # GCS has TSG type
                    gcs_type = 'TSG'

    # GCS tagged w/ type
    data_gcs['hasType'] = gcs_type


def prepare_sent_data(item):
    """
    Prepare sentence data and return structured dict

    :param item: target sentence to ingest
    :return: target sentence as structured dict
    """

    sent = {
        'Content': item['Sentence'],
        'Curated': True if item['Curated'] else False,
        'GeneStart': item['GeneStart'], 'GeneEnd': item['GeneEnd'], 'GeneMention': item['Sentence'][item['GeneStart']:item['GeneEnd']],
        'DiseaseStart': item['DiseaseStart'], 'DiseaseEnd': item['DiseaseEnd'], 'DiseaseMention': item['Sentence'][item['DiseaseStart']:item['DiseaseEnd']],
        'CGE': {'Label': item['CGE_label'], 'Score': item['CGE_score']},
        'CCS': {'Label': item['CCS_label'], 'Score': item['CCS_score']},
        'GCI': {'Label': item['GCI_label'], 'Score': item['GCI_score']},
        'GCC': {'Label': item['GCC_label'], 'Score': item['GCC_score']},
        'Database': src2db[item['OriginalDB']] if item['OriginalDB'] in src2db else '',
        'Paper': {'PMID': int(item['PMID']), 'PMIDYear': int(item['PMIDYear']), 'PMIDVenue': item['PMIDVenue'], 'PMIDTitle': item['PMIDTitle'], 'PMIDAbstract': item['PMIDAbstract']}
        }

    return sent