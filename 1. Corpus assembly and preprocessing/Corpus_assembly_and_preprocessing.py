# -*- coding: utf-8 -*-
"""
Spyder Editor
Created on Tue Mar 23 19:14:07 2021
@author: Francis Lareau
This is Projectjstor.
Corpus assembly and preprocessing.
"""

#==============================================================================
# ############################################################## Import library
#==============================================================================

import os
import sys
import pandas as pd
import pickle
import datetime
import re
import bz2
from sklearn.feature_extraction.text import CountVectorizer
import treetaggerwrapper #  TreeTagger must be installed and path specified
treetagger_path = "C:\TreeTagger"

#==============================================================================
# #################################################### Initialize project paths
#==============================================================================

main_path = os.path.join("your_main_path")
os.chdir(main_path)

#==============================================================================
# ################################################################# Import data
#==============================================================================

# For legal issues, the complete full-text of journal articles could not be 
# included with the dataset (but can be retrieved by asking JSTOR and the 
# respective publishers).

dataframe = pd.read_pickle(os.path.join(main_path,
                                        "0. Data",
                                        "Private",
                                        "DataFrame_Consolidation_updated_v2.pkl"))

dataframe_ne = pd.read_pickle(os.path.join(main_path,
                                           "0. Data",
                                           "Private",
                                           "DataFrame_Consolidation_updated_notenglish.pkl"))

#==============================================================================
# ################################################################# Filter Data
#==============================================================================

dataframe['translated']=False

for i in range(len(dataframe_ne)):
    dataframe.loc[dataframe_ne['index'][i],'Article'] = dataframe_ne.loc[i].Article
    dataframe.loc[dataframe_ne['index'][i],'Statut'] = 'OUI'
    dataframe.loc[dataframe_ne['index'][i],'translated'] = True

# retain article with positive status
dataframe=dataframe[dataframe.Statut=='OUI'] #exclude 'NON'
dataframe=dataframe[dataframe.Year!='2018'] #exclude '2018'
dataframe=dataframe[dataframe.Journal_id!='BP']
dataframe=dataframe[dataframe.Journal_id!='SHPSB']
dataframe=dataframe[dataframe.Journal_id!='SHPSC']
dataframe.reset_index(inplace=True)

#==============================================================================
# ################################################################## Clean Data
#==============================================================================

dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '\s\.','.',x,flags=re.DOTALL)) #fix space dot
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '(?<=[^\d])\.\d+','.',x,flags=re.DOTALL)) #fix note # after dot
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        'et\sal\.','et al',x,flags=re.DOTALL)) #erase dot of et al.
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '(?<=[\s\(\[])p\.\s','p.',x,flags=re.DOTALL)) #fix page dot
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '(?<=[a-z])\.(?=[A-Z])','. ',x,flags=re.DOTALL)) #fix dot space
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '[^\s]*@[^\s]*',' ',x,flags=re.DOTALL)) #erase email
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        'http[^\s]+',' ',x,flags=re.DOTALL)) #erase web page
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '[^A-z\-\s\'\"\.\?\!\,;:\(\)\[\]&\dàáâæçèéêëìíîïñòóôœùúûüýÿÀÁÂÆÇÈÉÊËÌÍÎÏÑÒÓÔŒÙÚÛÜÝŸ]',
        ' ',x,flags=re.DOTALL)) #erase all non word character
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '-{2,}','-',x,flags=re.DOTALL)) #erase multiple '-'
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '(?<=\s)-(?=[A-z])',' ',x,flags=re.DOTALL)) #erase '-' before word
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '\\\\',' ',x,flags=re.DOTALL)) #erase '\'
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '\^',' ',x,flags=re.DOTALL)) #erase '^'
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '_',' ',x,flags=re.DOTALL)) #erase '_'
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '[\d\.\,]*\d[\d\.\,]*',' ',x,flags=re.DOTALL)) #erase all digit
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '(?<=[\s\(\)\.\,\!\?:;])[\s\-]+',' ',x,flags=re.DOTALL)) #erase '\s\-' after
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '[\s\-]+(?=[\s\(\)\.\,\!\?:;])',' ',x,flags=re.DOTALL)) #erase '\s\-' before
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '(?<=[A-z])\.+(?=[A-z])',' ',x,flags=re.DOTALL)) #erase dot in word
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '\.{3,}',' ',x,flags=re.DOTALL)) #erase triple dot
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '[\.\-]+(?=[A-z])',' ',x,flags=re.DOTALL)) #erase '\.\-' before word
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '(?<=[A-z])\.+\-+(?=[A-z])',' ',x,flags=re.DOTALL)) #erase '\.\-' in word
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '(?<=[A-z])\-+\.+(?=[A-z])',' ',x,flags=re.DOTALL)) #erase '\-\.' in word
dataframe['Article']=dataframe['Article'].apply(lambda x: re.sub(
        '\s+',' ',x,flags=re.DOTALL)) #erase multiple space
    
#==============================================================================
# ############################################################## Clean Metadata
#==============================================================================
        
## FIX some volume of PS (Philosophy of Science)      
for i in range(len(dataframe)):
    if dataframe.Journal_id[i]=='PS':
        print(int(dataframe.Year[i])-1933)
        if dataframe.Issue[i]=='':
            dataframe.Volume[i]=dataframe.Year[i]
            dataframe.Issue[i]='proceedings'
        else:
            dataframe.Volume[i]=str(int(dataframe.Year[i])-1933)
        
## IX some volume of  BJPS (supplement);
for i in range(1281,1289):
    print(dataframe.Issue[i])
    dataframe.Issue[i]='S1'
        
#==============================================================================
# ############################################################### Make Citation
#==============================================================================

for idx,row in dataframe.iterrows():
    #author=''
    for idx_,(a,b) in enumerate(row.Author):        
        if idx_==0:
            author=a+', '+b 
        elif idx_!=0 and idx_!=len(row.Author)-1:
            author=author+', '+b+' '+a+','        
        else: #if idx_!=0 and idx_==len(row.Author)-1:
            author=author+' and '+b+' '+a
        author=re.sub('\.$','',author)
        Title=re.sub('\s+$','',row.Title)
        if Title[-1]!='.' and Title[-1]!='?' and Title[-1]!='!':
            Title=Title+'.'
        Journal = row.Journal_id
        if Journal=='BJPS':
            Journal='The British Journal for the Philosophy of Science'
        elif Journal=='EJPS':
            Journal='European Journal for Philosophy of Science'
        elif Journal=='ERK':
            Journal='Erkenntnis'
        elif Journal=='ISPS':
            Journal='International Studies in the Philosophy of Science'
        elif Journal=='JGPS':
            Journal='Journal for General Philosophy of Science'
        elif Journal=='PS':
            Journal='Philosophy of Science'
        elif Journal=='SHPSA':
            Journal='Studies in History and Philosophy of Science'
        elif Journal=='SYN':
            Journal='Synthese'
        else:
            print(idx, 'JOURNAL NAME ERROR')
    print(author+' ('+row.Year+')'+' '+Title+' '+Journal+' '+row.Volume+
          '('+row.Issue+'): '+row.Page_range+'.')
    dataframe['Citation'][idx]=(author+' ('+row.Year+')'+' '+Title+' '+Journal+' '+
                     row.Volume+'('+row.Issue+'): '+row.Page_range+'.')    
    
#==============================================================================
# ##################################################### Period and nb of author
#==============================================================================

dataframe['Period'] = DF_statistique_generale.Year.apply(lambda x: #22 years period
    str(int(x)-(int(x)-1908)%22)+'-'+str((int(x)-(int(x)-1908)%22)+21))

dataframe['nb_authors'] = DF_statistique_generale.Author.apply(lambda x: len(x))

#==============================================================================
# ############################################################## Fix authorship
#==============================================================================
            
with open(os.path.join(main_path,
                       "0. Data",
                       "author_to_correct.txt"), encoding='utf-8') as f:
    author_to_correct = [tuple(x.strip().split(', ',1)) for x in f]

for idx, x in enumerate(author_to_correct):
    if x == ('Skipper', 'Jr., Robert A.'):
        author_to_correct[idx]=('Skipper Jr.', 'Robert A.')
    if x == ('Godfrey-Smith', 'Peter'):
        author_to_correct[idx]=(' Godfrey-Smith', 'Peter')
    if x == ('Hitchcock', 'Christopher'):
        author_to_correct[idx]=(' Hitchcock', 'Christopher')
            
        
with open(os.path.join(main_path,
                       "O. Data",
                       "author_corrected.txt"), encoding='utf-8') as f:
    author_corrected = [tuple(x.strip().split(', ',1)) for x in f]

author_dict = dict(zip(author_to_correct, author_corrected))

for idx, group_author in enumerate(dataframe.Author):
    group_author_fixed=list()
    for author in group_author:
        if author in author_to_correct:
            group_author_fixed.append(author_dict[author])
        else:
            group_author_fixed.append(author)
    dataframe.Author[idx]=group_author_fixed            

#==============================================================================
# ############################################################### Save Metadata
#==============================================================================

dataframe[["Journal_id","Title", "Author", "Year","Volume","Issue","Page_range",
           "Citation","Article_ID","Lang_detect_1","Lang_detect_2",
           "translated",'Period','nb_authors']].to_pickle(
           os.path.join(main_path,
                        "0. Data",
                        "DF_philosophy_of_science_all_metadata_v2.pkl"))

#==============================================================================
# ########################### Word tokenization, POS tagging, and lemmatization
#==============================================================================

tagger = treetaggerwrapper.TreeTagger(TAGLANG='en', TAGDIR=treetagger_path)
dataframe['Lemma'] = ''

time_start = datetime.datetime.now()
for i in range(len(dataframe)):
    tokens = treetaggerwrapper.make_tags(tagger.tag_text(dataframe.Article[i]))
    list_of_lemma=[]
    for token in tokens:
        if (isinstance(token,treetaggerwrapper.Tag)
        and bool(re.findall('FW|MD|VVPRHASAL|VV.?|JJ.?|NN.?|NP.?|RB.?',token[1]))        
        and len(token[2])>=3):            
            token = token[2].lower()
            token = re.sub('^[\.\-\s\']+|[\.\-\s\']+$|^[a-z][\-\'][a-z]$','',token)
            list_of_lemma.append(token)
    dataframe.Lemma[i]=list_of_lemma
    sys.stdout.write("\rLemmatizing till article #"+str(i)+" took %s"%(str(datetime.datetime.now() - time_start)))
    sys.stdout.flush() # 3h

#==============================================================================
# ############################################################### Vectorization
#==============================================================================

def identity_tokenizer(text):
    ''' Method to use with Countvectorizer '''
    return text

#building stopwords set
stopwords = {line.strip() for line in open(
        os.path.join(main_path,
                     "0. Data",
                     "stopwords_en.txt"),
                     encoding='utf-8')}
rarewords = {line.strip() for line in open(
        os.path.join(main_path,
                     "0. Data",
                     "rarewords_en.txt"),
                     encoding='utf-8')} #based on min_df = 50 from sentences
stopwords = stopwords.union(rarewords)
stopwords = stopwords.union({'','etc.'})

#setup
vectorizer = CountVectorizer(lowercase = False,
                             min_df = 0,
                             analyzer = 'word', 
                             tokenizer = identity_tokenizer,
                             preprocessor = identity_tokenizer,
                             stop_words = stopwords, 
                             ngram_range = (1, 1))
#Create matrix and vocab
freq_term_matrix = vectorizer.fit_transform(dataframe.Lemma)
vocab = vectorizer.vocabulary_

#==============================================================================
# ############################################# Save Data (matrix & vocabulary)
#==============================================================================

with bz2.BZ2File(os.path.join(main_path,
                              "0. Data",
                              "DTM_philosophy_of_science_all.pbz2"), "w") as f:
    pickle.dump(freq_term_matrix, f, pickle.HIGHEST_PROTOCOL)
    
with open(os.path.join(main_path,
                       "0. Data",
                       "Vocabulary_philosophy_of_science_all.pkl"), "wb") as f:
    pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
