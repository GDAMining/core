# CORE
This repository contains the source code of the Collaborative Oriented Relation Extraction (CORE) system for building Knowledge Bases (KBs) with limited manual annotations. CORE is based on the combination of distant supervision and active learning paradigms, and features a seamless, transparent, modular architecture suited to large-scale processing. 

*This work has been submitted to KDD 2023 and is currently under review.*

## Code

### Source Code

The source code to build the CORE system can be found in ```./src```.
The code divides into:
- ```./src/core```: contains the code required to build the KB
- ```./src/data```: contains the code required to process data
- ```./src/models```: contains the code required to make RE models

### Data Processing

The scripts used to process data divide into:
- ```retrieve_pubmed_data.py``` and ```crawl_citing_papers.py``` which gather PubMed data
- ```process_disgenet_data.py``` and ```process_pubtator_data.py``` which process DisGeNET and PubTator data
- ```prepare_data.py``` which prepares data for KB construction

### Relation Extraction

The scripts used for RE models can be grouped in three categories: training, evaluation, deployment.

Training:
- ```train_cebert.py``` can be used to train aspect extractors (CGE, CCS, GCI)
- ```train_cerefiner.py``` can be used to train the sentence utility classifier (GCC)

Evaluation:
- ```evaluate_cebert.py``` can be used to evaluate aspect extractors (CGE, CCS, GCI)
- ```evaluate_cerefiner.py``` can be used to evaluate the sentence utility classifier (GCC)

Deployment:
- ```deploy_cebert.py``` can be used to deploy aspect extractors (CGE, CCS, GCI)
- ```deploy_cerefiner.py``` can be used to deploy the sentence utility classifier (GCC)

### Knowledge Base Construction

To build the KB, use the ```build_cecore.py``` script.

## Knowledge Base

The Gene Expression-Cancer KB derived by CORE is available as Open Data at: https://doi.org/10.5281/zenodo.7577127.  <br />
The repository consists of the ```schema.owl``` file, which contains the KB schema, and the ```data.ttl``` file, which contains the KB data.

## SPARQL End-Point

The KB can be accessed via a SPARQL end-point, available at: http://w3id.org/corekb/sparql. <br />
The ```./sparql/example-queries.txt``` file lists example queries that can be used on the KB to compute statistics, explore data, and discover trends. Queries can be copy-pasted into the SPARQL end-point and then executed.

## Acks
The work is supported by the [ExaMode project](https://www.examode.eu), as part of the EU H2020 program under Grant Agreement no. 825292.
The development of the CORE system benefited from the valuable contribution of the medical centres and clinicians involved in the ExaMode project. We would like to express our gratitude for their support and feedback, which has been instrumental in the creation of the CORE KB.
