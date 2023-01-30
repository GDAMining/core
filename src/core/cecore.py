import os
import hashlib as hash

from rdflib.namespace import RDF, XSD
from rdflib import Graph, Namespace, URIRef, Literal


class CECORE(object):
    """
    CE CORE KG
    """

    def __init__(self):
        # set CE CORE Graph
        self.graph = Graph()

        # set namespaces for CE CORE KB
        self.ns = {
            'ceonto': Namespace('http://gda.dei.unipd.it/cecore/ontology/'),
            'cesent': Namespace('http://gda.dei.unipd.it/cecore/resource/Sentence#'),
            'cegcs': Namespace('http://gda.dei.unipd.it/cecore/resource/GCS#'),
            'umls': Namespace('http://linkedlifedata.com/resource/umls/id/'),
            'ncbi': Namespace('https://www.ncbi.nlm.nih.gov/gene/'),
            'pubmed': Namespace('https://pubmed.ncbi.nlm.nih.gov/')
        }

        # bind namespaces to the given prefix
        for prefix, namespace in self.ns.items():
            self.graph.bind(prefix, namespace)

        # set GCS associations for CE CORE KB
        self.gcs2preds = {
            'gcs2gene': self.ns['ceonto']['expressedBy'],
            'gcs2cancer': self.ns['ceonto']['involves'],
            'gcs2sent': self.ns['ceonto']['supportedBy'],
            'gcs2type': self.ns['ceonto']['hasType']
        }

        # set Sentence associations for CE CORE KB
        self.sent2label = {
            'cge_label': self.ns['ceonto']['CGELabel'],
            'ccs_label': self.ns['ceonto']['CCSLabel'],
            'gci_label': self.ns['ceonto']['GCILabel'],
            'gcc_label': self.ns['ceonto']['GCCLabel']
        }

        self.sent2preds = {
            'sent2gene': self.ns['ceonto']['mentionsGene'],
            'sent2cancer': self.ns['ceonto']['mentionsDisease'],
            'text': self.ns['ceonto']['hasContent'],
            'gene_start': self.ns['ceonto']['GSAStartPosition'],
            'gene_end': self.ns['ceonto']['GSAEndPosition'],
            'gene_mention': self.ns['ceonto']['GSAMention'],
            'disease_start': self.ns['ceonto']['DSAStartPosition'],
            'disease_end': self.ns['ceonto']['DSAEndPosition'],
            'disease_mention': self.ns['ceonto']['DSAMention'],
            'curated': self.ns['ceonto']['curatedAnnotation'],
            'source': self.ns['ceonto']['extractedFrom'],
            'db': self.ns['ceonto']['includedIn']
        }

        # set Paper associations for CE CORE KB
        self.paper2preds = {
            'title': self.ns['ceonto']['title'],
            'abstract': self.ns['ceonto']['abstract'],
            'venue': self.ns['ceonto']['hasVenue'],
            'year': self.ns['ceonto']['publicationYear'],
            'cites': self.ns['ceonto']['cites'],
            'has_cites': self.ns['ceonto']['hasCitation']
        }

        # set ontology entities
        self.ents = {
            'gene': URIRef('http://purl.obolibrary.org/obo/SO_0000704'),
            'disease': URIRef('https://www.ebi.ac.uk/ols/ontologies/doid/Disease'),
            'gcs': URIRef('http://gda.dei.unipd.it/cecore/ontology/GCS'),
            'sent': URIRef('http://semanticscience.org/resource/SIO_000113'),
            'db': URIRef('http://purl.obolibrary.org/obo/NCIT_C15426'),
            'paper': URIRef('http://purl.obolibrary.org/obo/NCIT_C47902')
        }

    def convert_gcs2rdf(self, gene_cancer, gcs):
        """
        Convert GCS data to RDF format

        :param gene_cancer: GCS gene/cancer pair
        :param gcs: GCS data
        :return: True if conversion has been performed
        """

        # get gene and cancer IDs and convert to URIs
        entrez_id = str(gene_cancer[0])
        cui = gene_cancer[1]

        gene_id = self.ns['ncbi'][entrez_id]
        cancer_id = self.ns['umls'][cui]

        # set gene and cancer associations between instances and entities
        self.graph.add((gene_id, RDF.type, self.ents['gene']))
        self.graph.add((cancer_id, RDF.type, self.ents['disease']))

        # set GCS ID as md5 hashing of the gene/cancer pair
        gcs_id = self.gcs2id(entrez_id, cui)

        # set GCS association between instance and entity
        self.graph.add((gcs_id, RDF.type, self.ents['gcs']))

        # set GCS associations with gene and cancer
        self.graph.add((gcs_id, self.gcs2preds['gcs2gene'], gene_id))
        self.graph.add((gcs_id, self.gcs2preds['gcs2cancer'], cancer_id))

        # set GCS association with sentences
        for sent in gcs['sents']:
            # set sentence ID
            sent_id = self.sent2id(sent)
            # set sentence association between instance and entity
            self.graph.add((sent_id, RDF.type, self.ents['sent']))
            # set GCS/sentence association
            self.graph.add((gcs_id, self.gcs2preds['gcs2sent'], sent_id))
            # set sentence/gene and sentence/disease associations
            self.graph.add((sent_id, self.sent2preds['sent2gene'], gene_id))
            self.graph.add((sent_id, self.sent2preds['sent2cancer'], cancer_id))
            # set sentence associations w/ literals
            self.convert_sent2rdf(sent_id, sent)

            # set paper ID
            paper_id = self.ns['pubmed'][str(sent['Paper']['PMID'])]
            # set paper association between instance and entity
            self.graph.add((paper_id, RDF.type, self.ents['paper']))
            # set sentence/paper association
            self.graph.add((sent_id, self.sent2preds['source'], paper_id))
            # set paper associations w/ literals
            self.convert_paper2rdf(paper_id, sent['Paper'])

        # set GCS association with GCS type
        self.graph.add((gcs_id, self.gcs2preds['gcs2type'], Literal(gcs['hasType'], datatype=XSD.string)))

        return True

    def gcs2id(self, gene_id, cancer_id):
        """
        Set GCS ID as md5 hashing gene and cancer IDs

        :param gene_id: gene ID
        :param cancer_id: cancer ID
        :return: hashed GCS ID
        """

        # set GCS ID as md5 hashing of the gene/cancer pair
        gcs_id = gene_id + cancer_id
        gcs_id = hash.md5(gcs_id.encode('utf-8')).hexdigest()
        gcs_id = self.ns['cegcs'][gcs_id]

        return gcs_id

    def sent2id(self, sent_data):
        """
        Set sentence ID as md5 hashing of the sentence data

        :param sent_data: sentence data
        :return: hashed sentence ID
        """

        # set sentence ID as md5 hashing of the sentence data
        sent_id = sent_data['Content']
        sent_id += str(sent_data['Curated'])
        sent_id += str(sent_data['GeneStart']) + str(sent_data['GeneEnd'])
        sent_id += str(sent_data['DiseaseStart']) + str(sent_data['DiseaseEnd'])
        sent_id += sent_data['Database']
        sent_id += str(sent_data['Paper']['PMID'])
        sent_id = hash.md5(sent_id.encode('utf-8')).hexdigest()
        sent_id = self.ns['cesent'][sent_id]

        return sent_id

    def convert_sent2rdf(self, sent_id, sent_data):
        """
        Convert sentence data to RDF format

        :param sent_id: sentence ID
        :param sent_data: sentence data
        :return: True if conversion has been performed
        """

        # CGE, CCS, GCI, and GCC associations
        self.graph.add((sent_id, self.sent2label['cge_label'], Literal(sent_data['CGE']['Label'], datatype=XSD.string)))
        self.graph.add((sent_id, self.sent2label['ccs_label'], Literal(sent_data['CCS']['Label'], datatype=XSD.string)))
        self.graph.add((sent_id, self.sent2label['gci_label'], Literal(sent_data['GCI']['Label'], datatype=XSD.string)))
        self.graph.add((sent_id, self.sent2label['gcc_label'], Literal(sent_data['GCC']['Label'], datatype=XSD.string)))
        # Content association
        self.graph.add((sent_id, self.sent2preds['text'], Literal(sent_data['Content'], datatype=XSD.string)))
        # Curated association
        self.graph.add((sent_id, self.sent2preds['curated'], Literal(sent_data['Curated'], datatype=XSD.boolean)))
        # Gene associations
        self.graph.add((sent_id, self.sent2preds['gene_start'], Literal(sent_data['GeneStart'], datatype=XSD.int)))
        self.graph.add((sent_id, self.sent2preds['gene_end'], Literal(sent_data['GeneEnd'], datatype=XSD.int)))
        self.graph.add((sent_id, self.sent2preds['gene_mention'], Literal(sent_data['GeneMention'], datatype=XSD.string)))
        # Disease associations
        self.graph.add((sent_id, self.sent2preds['disease_start'], Literal(sent_data['DiseaseStart'], datatype=XSD.int)))
        self.graph.add((sent_id, self.sent2preds['disease_end'], Literal(sent_data['DiseaseEnd'], datatype=XSD.int)))
        self.graph.add((sent_id, self.sent2preds['disease_mention'], Literal(sent_data['DiseaseMention'], datatype=XSD.string)))
        # Database associations -- if any
        if sent_data['Database']:
            self.graph.add((URIRef(sent_data['Database']), RDF.type, self.ents['db']))
            self.graph.add((sent_id, self.sent2preds['db'], URIRef(sent_data['Database'])))

        return True

    def convert_paper2rdf(self, paper_id, paper_data):
        """
        Convert paper data to RDF format

        :param paper_id: paper ID
        :param paper_data: paper data
        :return: True if conversion has been performed
        """

        # PMIDYear association
        if paper_data['PMIDYear'] > 0:
            self.graph.add((paper_id, self.paper2preds['year'], Literal(paper_data['PMIDYear'], datatype=XSD.int)))
        # PMIDVenue association
        if paper_data['PMIDVenue']:
            self.graph.add((paper_id, self.paper2preds['venue'], Literal(paper_data['PMIDVenue'], datatype=XSD.string)))
        # PMIDTitle association
        if paper_data['PMIDTitle']:
            self.graph.add((paper_id, self.paper2preds['title'], Literal(paper_data['PMIDTitle'], datatype=XSD.string)))
        # PMIDAbstract association
        if paper_data['PMIDAbstract']:
            self.graph.add((paper_id, self.paper2preds['abstract'], Literal(paper_data['PMIDAbstract'], datatype=XSD.string)))

        return True

    def serialize(self, outf):
        """
        Serialize CE CORE KG

        :param outf: output file
        :return: True if serialization has been performed
        """

        # create output dir if not exist
        os.makedirs(os.path.dirname(outf), exist_ok=True)
        # serialize CE CORE into RDF format (turtle)
        print('Serialize CE CORE KB to {} with turtle format...'.format(outf))
        self.graph.serialize(destination=outf, format='turtle')
        print('CE CORE KB serialized to {} with turtle format!'.format(outf))
        return True
