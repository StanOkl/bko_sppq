## Replication files for Butler, Kousser and Oklobdzija (2022). 


"Do Male and Female Legislators have Different Twitter Communication Styles?"

By Dan Butler, Thad Kousser and Stan Oklobdzija -- Washington University in St. Louis, University of California, San Diego and University of California, Riverside -- Corresponding Author: Thad Kousser, tkousser@ucsd.edu. 

Replication files hosted at https://github.com/StanOkl/bko_sppq.

### Software: 

Machine Learning component conducted with Python 3.7.2.

Statistical analyses and figures produced with R 4.1.2.

### Analysis:

All ML coding and analyses can be produced by running bko_replication_sppq.R. This code is also annotated to describe the various steps involved in producing each figure and table. 

### List of files:

#### Data Files:

CodedStateTweetsMarch2020.xlsx -- Dataset of 1,898 tweets from state legislators coded by human research assistants.

Lower_House_Tweeting_Full with Ideal Points.dta -- Data to produce regression tables, (Tables 4:8), and Figure 1.

State Legislator Twitter Handles - Lower House.csv -- Names and Twitter handles for 2,795 members of various State Legislatures.

State_Leg_Tweets_corpus.csv -- Full corpus of 3,580,727 tweets, unclassified and unprocessed. 

ProcessedFullCorpus_state.csv -- Full corpus of 3,580,727 tweets, unclassified and processed.

Training_Tweets_8-9.csv -- 7,619 2016 Presidential tweets classified by two coders used to calculate inter-rater reliability statistics for Table 1. 

training_data.csv -- Dataset of 8,206 tweets from Presidential candidates in 2016 coded by human research assistants.

training_data_stateprez_processed.csv -- Combined 10,105 human coded tweets from 2016 Presidential campaign and state legislators used to train ML algorithms used to classify full corpus. 

#### Code: 

bko_replication_script.R -- Master script to run ML classification and produce tables and figures from paper. 

ml_classifier_final.py -- ML script to produce accuracy measures for each category and then classify full corpus of tweets. 

#### Documentation 

README.md -- Description of code and data for article. 

codebook.pdf -- Codebook for data used to produce tables and figures.
