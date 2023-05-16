# -*- coding: utf-8 -*-
"""
Spyder Editor
Created on Tue Mar 30 20:41:42 2021
@author: Francis Lareau
This is Projectjstor.
Author topic profiles.
"""

#==============================================================================
# ############################################################## Import library
#==============================================================================

import os
import sys
import pandas as pd
import numpy as np
import pickle
import bz2

#==============================================================================
# #################################################### Initialize project paths
#==============================================================================

main_path = os.path.join("your_main_path")
os.chdir(main_path)

#==============================================================================
# ################################################################# Import data
#==============================================================================

with open(os.path.join(main_path,
                       "0. Data",
                       "LDA_model_philosophy_of_science_all_K25.pkl"), "rb") as f:
    ldamodel_lda = pickle.load(f)

DTM = pd.read_pickle(bz2.BZ2File(
        os.path.join(main_path,
                     "0. Data",
                     "DTM_philosophy_of_science_all.pbz2"), 'rb'))

with open(os.path.join(main_path,
                       "0. Data",
                       "Vocabulary_philosophy_of_science_all.pkl"), "rb") as f:
    Vocab = pickle.load(f)
    
DF_statistique_generale = pd.read_pickle(
        os.path.join(main_path,
                     "0. Data",
                     "DF_philosophy_of_science_all_metadata_v2.pkl"))

#==============================================================================
# ##################### Data statistic, lda model score and lda hyperparameters
#==============================================================================
  
df_param=pd.DataFrame(index=['Value'])
df_param['Sparsity']=((DTM.todense() > 0).sum() / 
        DTM.todense().size*100) #sparsicity (% nonzero)
df_param['Log Likelyhood']=ldamodel_lda.loglikelihood() #Log Likelyhood (higher better)
df_param['Perplexity']='' #Perplexity (lower better, exp(-1. * log-likelihood per word)
df_param['alpha']=ldamodel_lda.alpha
df_param['eta']=ldamodel_lda.eta
df_param['n_iter']=ldamodel_lda.n_iter
df_param['n_components']=ldamodel_lda.n_topics
df_param['random_state']=ldamodel_lda.random_state
df_param['refresh']=ldamodel_lda.refresh

#==============================================================================
# ########################################################### Topic by document
#==============================================================================

#Topic for each document
lda_output=ldamodel_lda.doc_topic_
topicnames = ["Topic_" + str(i) for i in range(len(ldamodel_lda.components_))]
docnames = [i for i in range(DTM.shape[0])]
df_document_topic = pd.DataFrame(lda_output, 
                                 columns=topicnames,
                                 index=docnames)
dominant_topic = np.argmax(df_document_topic.values, axis=1)
#add results to statistic general
DF_statistique_generale['Dom_topic'] = dominant_topic
DF_topic=pd.concat([DF_statistique_generale,df_document_topic],
                   axis=1,
                   join='inner')
    
#count document by topic
df_topic_distribution = DF_statistique_generale['Dom_topic'].value_counts(
        ).reset_index(name="Num_Documents")
df_topic_distribution.columns = ['Topic_Num', 'Num_Doc']
# Topic - keyword Matrix
df_topic_keywords = pd.DataFrame(ldamodel_lda.components_)#every row =1
df_topic_keywords.index = topicnames
#Transpose to topic - keyword matrix
df_keywords_topic = df_topic_keywords.transpose()
df_keywords_topic.index = sorted([i for i in Vocab.keys()])
# Topic - Top Keywords Dataframe
n_top_words = 50+1
DF_Topic_TKW = pd.DataFrame(columns=range(n_top_words-1),index=range(len(ldamodel_lda.components_)))
vocab = sorted([i for i in Vocab.keys()])
topic_word = ldamodel_lda.components_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    DF_Topic_TKW.loc[i]=topic_words

DF_Topic_TKW.columns = ['Word_'+str(i) for i in range(DF_Topic_TKW.shape[1])]
DF_Topic_TKW.index = ['Topic_'+str(i) for i in range(DF_Topic_TKW.shape[0])]
DF_Topic_TKW['Sum_Doc'] = np.array(DF_statistique_generale['Dom_topic'].value_counts(
        ).sort_index())
DF_Topic_TKW['Top-10_Words'] = ''
for idx,row in DF_Topic_TKW.iterrows():
    DF_Topic_TKW['Top-10_Words'][idx]=(row['Word_0']+'; '+row['Word_1']+'; '+
                row['Word_2']+'; '+row['Word_3']+'; '+row['Word_4']+'; '+
                row['Word_5']+'; '+row['Word_6']+'; '+row['Word_7']+'; '+
                row['Word_8']+'; '+row['Word_9'])

#==============================================================================
# ############################################################# Topic by author
#==============================================================================       

# Author - Topic Matrix
authors = set()
authors_list = list()
for group_author in DF_statistique_generale.Author:
    for author in group_author:
        authors.add(author)
        authors_list.append(author)
authors = sorted(authors)
        
DF_AT = pd.DataFrame(data='', index=range(len(authors)),columns=topicnames+['Author','Pub_sum','Pub_weighted','Pub_w_1','Pub_w_2','Pub_w_3','Pub_w_4'])
DF_AT_norm = pd.DataFrame(data='', index=range(len(authors)),columns=topicnames)
for idx,author in enumerate(authors):
    list_bool = DF_statistique_generale.Author.apply(lambda x: True if author in x else False)
    author_topic=sum(lda_output[list_bool]/np.transpose(np.repeat(np.expand_dims(DF_statistique_generale.nb_authors[list_bool],axis=0), repeats=25, axis=0)))
    author_topic_norm=author_topic/sum(author_topic)
    DF_AT.loc[idx]=list(author_topic)+[author,
             authors_list.count(author),
             sum(1/DF_statistique_generale.nb_authors[list_bool]),
             sum(1/DF_statistique_generale.nb_authors[list_bool & DF_statistique_generale.Period.isin(['1930-1951'])]),
             sum(1/DF_statistique_generale.nb_authors[list_bool & DF_statistique_generale.Period.isin(['1952-1973'])]),
             sum(1/DF_statistique_generale.nb_authors[list_bool & DF_statistique_generale.Period.isin(['1974-1995'])]),
             sum(1/DF_statistique_generale.nb_authors[list_bool & DF_statistique_generale.Period.isin(['1996-2017'])])]
    DF_AT_norm.loc[idx]=list(author_topic_norm)
 
#==============================================================================
# ################################################### Topic by period + author
#==============================================================================

# Topic - Journal + Period Matrix (12hre)
DF_PAT = pd.DataFrame([[item for sublist in [sorted(set(DF_topic.Period))]*len(authors) for item in sublist],
                         [item for sublist in [[x]*len(set(DF_topic.Period)) for x in authors] for item in sublist]
                         ], index=['Period','Author']).transpose()
DF_PAT=DF_PAT.reindex(columns =['Period','Author']+topicnames)
for idx,row in DF_PAT.iterrows():    
    sys.stdout.write("\r"+str(idx)+"/"+str(len(DF_PAT))) # \r prints a carriage return, then we print on top of the previous line.
    list_bool_1 = DF_topic.Author.apply(lambda x: True if row.Author in x else False)
    list_bool_2 = DF_topic.Period.apply(lambda x: True if row.Period in x else False)
    if sum(list_bool_1 & list_bool_2):
        author_period_topic = sum(lda_output[list_bool_1 & list_bool_2]/np.transpose(np.repeat(np.expand_dims(DF_statistique_generale.nb_authors[list_bool_1 & list_bool_2],axis=0), repeats=25, axis=0)))
        DF_PAT.loc[idx]=[row[0],row[1]]+list(author_period_topic)
        
#==============================================================================
# ########################################################## Author correlation
#==============================================================================

# Author Pearson Correlation
DF_AfromT = DF_AT_norm.astype('float64').T.corr(method='pearson')

Pub_weighted = DF_AT.Pub_weighted > 2
DF_AfromT_Pub_weighted_1 = DF_AT_norm[Pub_weighted].astype('float64').T.corr(method='pearson')
DF_AfromT_Pub_weighted_2 = DF_AfromT_Pub_weighted_1.copy()
for i in range(len(DF_AfromT_Pub_weighted_1)):
    DF_AfromT_Pub_weighted_2.iloc[i]=[0 if (e <= .7) else e for e in DF_AfromT_Pub_weighted_2.iloc[i]]

DIC_DF_ATP = dict()
for period_name in set(DF_statistique_generale.Period):
    DIC_DF_ATP[period_name] = pd.DataFrame(columns=topicnames)
    for idx,author in enumerate(authors):
        list_bool = DF_statistique_generale[DF_statistique_generale.Period.isin([period_name])].Author.apply(lambda x: True if author in x else False)
        if lda_output[list_bool & DF_statistique_generale.Period.isin([period_name])].size > 0 :
            author_topic=sum(lda_output[list_bool & DF_statistique_generale.Period.isin([period_name])]/np.transpose(np.repeat(np.expand_dims(DF_statistique_generale.nb_authors[list_bool & DF_statistique_generale.Period.isin([period_name])],axis=0), repeats=25, axis=0)))
            author_topic_norm=author_topic/sum(author_topic)
            DIC_DF_ATP[period_name].loc[idx]=list(author_topic_norm)

DIC_DF_ATP_corr = dict()
for period_name in set(DF_statistique_generale.Period):
    DIC_DF_ATP_corr[period_name] = DIC_DF_ATP[period_name].astype('float64').T.corr(method='pearson')

for period_name in set(DF_statistique_generale.Period):
    file_path = os.path.join(main_path,
                             "0. Data",
                             "2. Data analysis and HCols identification",
                             "2.2 Author topic profiles",
                             "DF_Authors_corr_"+period_name+"_plain_01_more.csv")
    DIC_DF_ATP_corr[period_name].where(lambda x: x >= 0.1, '').to_csv(file_path,encoding="utf-8")
      
#==============================================================================
# ################################################################ Save results
#==============================================================================
        
# Save lda results to excel
writer = pd.ExcelWriter(os.path.join(main_path,
                                     "2. Data analysis and HCols identification",
                                     "2.2 Author topic profiles",
                                     "Results_from_diachronic_and_author_analyses.xlsx"))
df_param.T.to_excel(writer,'Para Score',encoding='utf8')      
DF_topic.to_excel(writer,'Doc vs Topic',encoding='utf8')
DF_AT.to_excel(writer,'Authors vs Topics',encoding='utf8')
DF_PAT.to_excel(writer,'Authors+P vs Topics',encoding='utf8')
writer.save()
writer.close()
