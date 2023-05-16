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
                     "DF_philosophy_of_science_all_metadata.pkl"))

DF_statistique_generale['Article'] = pd.read_pickle( #this one include articles that we can't publicly share
        os.path.join(main_path,
                     "0. Data",
                     "Private",
                     "DataFrame_Consolidation_updated_v2_notenglish_General_Stat_from_Tagged_v3.pkl")).Article
        
DF_statistique_generale['Period'] = DF_statistique_generale.Year.apply(lambda x: #22 years period
    str(int(x)-(int(x)-1908)%22)+'-'+str((int(x)-(int(x)-1908)%22)+21))

DF_statistique_generale['nb_authors'] = DF_statistique_generale.Author.apply(lambda x: len(x))

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
# ############################################################# Topic by period
#==============================================================================

# Topic - Period Matrix
DF_PT=pd.DataFrame(lda_output,
                   columns=topicnames,
                   index=docnames)

DF_PT['Period']=DF_topic.Period
DF_PT = DF_PT.groupby(['Period']).sum()
DF_TP = DF_PT.transpose()
DF_TP = DF_TP/DF_TP.sum()
DF_TP_Overall = DF_PT.transpose()
DF_TP_Overall['Raw'] = DF_PT.sum()
DF_TP_Overall['Overall'] = DF_PT.sum() / sum(DF_PT.sum())

# Periods - Topics top_10 articles Matrix (sorted by year)
DF_PT_T10A=pd.DataFrame(data='', index=DF_TP.columns,columns=DF_TP.index)
for period in DF_TP.columns:
    for topic in DF_TP.index:
        for idx in DF_topic[DF_topic.Period==period].nlargest(
                10,topic).sort_values('Year',ascending=False).index:
            DF_PT_T10A[topic][period]=DF_PT_T10A[topic][period]+DF_topic.Citation[idx]+'\n'
            
# Topics top_20 articles Matrix by Periods - (sorted by weight)
DF_PT_T20A=pd.DataFrame(data='', index=DF_TP.columns,columns=DF_TP.index)
for period in DF_TP.columns:
    for topic in DF_TP.index:
        for idx in DF_topic.nlargest(20,topic).index:
            if DF_topic.Period[idx]==period:
                DF_PT_T20A[topic][period]=DF_PT_T20A[topic][period]+DF_topic.Citation[idx]+'\n'

#==============================================================================
# ############################################################## Fix authorship
#==============================================================================
            
with open(os.path.join(main_path,
                       "Diachronic and author analysis",
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
                       "Diachronic and author analysis",
                       "author_corrected.txt"), encoding='utf-8') as f:
    author_corrected = [tuple(x.strip().split(', ',1)) for x in f]

author_dict = dict(zip(author_to_correct, author_corrected))

for idx, group_author in enumerate(DF_statistique_generale.Author):
    group_author_fixed=list()
    for author in group_author:
        if author in author_to_correct:
            group_author_fixed.append(author_dict[author])
        else:
            group_author_fixed.append(author)
    DF_statistique_generale.Author[idx]=group_author_fixed            

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
    #author_topic=sum(lda_output[list_bool])/len(lda_output[list_bool])
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
# ################################################################ Save results
#==============================================================================
        
# Save lda results to excel
writer = pd.ExcelWriter(os.path.join(main_path,
                                     "Diachronic and author analysis",
                                     "Results_from_diachronic_and_author_analyses_v9.xlsx"))
df_param.T.to_excel(writer,'Para Score',encoding='utf8')    
#pd.DataFrame([Vocab,[x[0] for x in DTM.T.sum(axis=1).tolist()]]).T.to_excel(writer,'Vocab',encoding='utf8')    
#DF_topic.to_excel(writer,'Doc vs Topic',encoding='utf8')
#DF_Topic_TKW.to_excel(writer,'Top 50 Topics Words',encoding='utf8')
#DF_TKW_cor.to_excel(writer,'Top 10 Correlates',encoding='utf8')
#df_keywords_topic.to_excel(writer,'Words vs Topics',encoding='utf8',
#                           header=topicnames,
#                           index=sorted([i for i in Vocab.keys()]))
 DF_AT.to_excel(writer,'Authors vs Topics',encoding='utf8')
#DF_CT.to_excel(writer,'Cluster mean vs Topics',encoding='utf8')
#DF_CT_center.to_excel(writer,'Cluster center vs Topics',encoding='utf8')
 DF_PAT.to_excel(writer,'Authors+P vs Topics',encoding='utf8')
#DF_TP.to_excel(writer,'Topics vs Periods',encoding='utf8')
#DF_TP_Overall.to_excel(writer,'Overall Topics vs Periods',encoding='utf8')
#DF_PT_T10A.to_excel(writer,'Top 10 articles',encoding='utf8')
#DF_PT_T20A.to_excel(writer,'Top 20 articles',encoding='utf8')
#DF_TfromD.to_excel(writer,'Topic Cor. from Doc',encoding='utf8')
#DF_TfromW.to_excel(writer,'Topic Cor. from Word',encoding='utf8')
#DF_AfromT.to_excel(writer,'Author Cor.',encoding='utf8')
writer.save()
writer.close()
