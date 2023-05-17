# CORE
This repository contains the source code of the Collaborative Oriented Relation Extraction (CORE) system for building Knowledge Bases (KBs) with limited manual annotations. CORE is based on the combination of distant supervision and active learning paradigms, and features a seamless, transparent, modular architecture suited to large-scale processing. 

## Code & Scripts

### Source Code

The source code to build the CORE system can be found in ```./src```.
The code divides into:
- ```./src/core```: contains the code required to build the KB
- ```./src/data```: contains the code required to process data
- ```./src/models```: contains the code required to make RE models

### Data Processing

The scripts used to process data divide into:
- ```./data_proc/process_disgenet_data.py```: processes DisGeNET data
- ```./data_proc/crawl_citing_papers.py```: crawls citing PubMed papers
- ```./data_proc/retrieve_pubmed_data.py```: retrieves missing PubMed data
- ```./data_proc/process_pubtator_data.py``` processes PubTator data

### Relation Extraction

The scripts used for RE models can be grouped in three categories: training, evaluation, deployment.

Training:
- ```./train/train_gca.py``` can be used to train gene expression-cancer aspect extractors (CGE, CCS, GCI)
- ```./train/train_gcc.py``` can be used to train the sentence utility classifier (GCC)

Evaluation:
- ```./eval/eval_gca.py``` can be used to evaluate gene expression-cancer aspect extractors (CGE, CCS, GCI)
- ```./eval/eval_gcc.py``` can be used to evaluate the sentence utility classifier (GCC)

Deployment:
- ```./deploy/deploy_gca.py``` can be used to deploy gene expression-cancer aspect extractors (CGE, CCS, GCI)
- ```./deploy/deploy_gcc.py``` can be used to deploy the sentence utility classifier (GCC)

### Knowledge Base Construction

To build the KB, first prepare the data with ```prepare_data.py``` and then use the ```build_kb.py``` script.

## Knowledge Base

The Gene Expression-Cancer KB derived by CORE is available as Open Data on Zenodo: https://doi.org/10.5281/zenodo.7577127.  <br />
The repository consists of the ```schema.owl``` file, which contains the KB schema, and the ```data.ttl``` file, which contains the KB data.

## SPARQL End-Point

The KB can be accessed via a SPARQL end-point, available at: http://w3id.org/corekb/sparql. <br />
The ```./sparql/example-queries.txt``` file lists example queries that can be used on the KB to compute statistics, explore data, and discover trends. Queries can be copy-pasted into the SPARQL end-point and then executed.

## Acknowledgments
The work is supported by the [ExaMode project](https://www.examode.eu), as part of the EU H2020 program under Grant Agreement no. 825292.
The development of the CORE system benefited from the valuable contribution of the medical centres and clinicians involved in the ExaMode project. We would like to express our gratitude for their support and feedback, which has been instrumental in the creation of the CORE KB.
