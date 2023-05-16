# Inferring_social_networks_from_unstructured_text_data
## Abstract
Social network analysis is known to provide a wealth of insights relevant to many aspects of policymaking. Yet the social data needed to construct social networks are not always available. Furthermore, even when they are, interpreting such networks often relies on extraneous knowledge. Here, we propose an approach to infer social networks directly from the texts produced by network agents and the terminological similarities that these texts exhibit. This approach relies on fitting a topic-model to the texts produced by these agents and measuring topic profile correlations between agents. This reveals what can be called “hidden communities of interest” (HCoI’s), that is to say groups of agents sharing similar semantic contents but whose social relationships with one another may be unknown or underlying. Network interpretation follows from the topic model. Diachronic perspectives can also be built by modeling the networks over different time-periods and mapping genealogical relationships between communities. As a case study, the approach is deployed over a working corpus of academic articles (domain of philosophy of science; <em>N</em>=16,917).

## Requirements
This code was tested on Python 3.7.3. Other requirements are as follows (see requirements.txt):
- bs4
- lda
- nltk
- numpy
- pandas
- spacy
- sklearn
- tmtoolkit
- treetaggerwrapper

## Quick Start
Install libraries: pip install -r requirements.txt

Install TreeTagger
### 1. Corpus assembly and preprocessing*
Execute to replicate research : Corpus_assembly_and_preprocessing.py
### 2. Data analysis and HCol's identification
#### 2.1 Topic-modeling
Execute to replicate research : Topic_modeling.py
#### 2.2 Author topic profiles
Execute to replicate research : Author_topic_profiles.py
#### 2.3 Author correlation networks
Done with Gephy
#### 2.4 HCol's distances
Execute to replicate research : HCols_distances.py

*Note that for legal issues, the complete full-text of journal articles could not be included with the dataset (but can be retrieved by asking the respective publishers).

## Citation
Malaterre, Christophe and Francis Lareau. 2023. Inferring social networks from unstructured text data: A proof of concept detection of “hidden communities of interest”. <em>Text and Data Analytics for Policy</em>.

## Authors
Christophe Malaterre
Email: malaterre.christophe@uqam.ca
Francis Lareau
Email: francislareau@hotmail.com
## Acknowledgments
CM acknowledges funding from Canada Social Sciences and Humanities Research Council (Grant 430-2018-00899) and Canada Research Chairs (CRC-950-230795). FL acknowledges funding from the Fonds de recherche du Québec Société et culture (FRQSC-276470) and the Canada Research Chair in Philosophy of the Life Sciences at UQAM. 
