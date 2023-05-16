# -*- coding: utf-8 -*-
"""
Spyder Editor
Created on Tue Mar 30 20:41:42 2021
@author: Francis Lareau
This is Projectjstor.
Data for diachronic and author analysis
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
# #################################### Top words correlations from weighted dtm
#==============================================================================

from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix

def compute_sparse_correlation_matrix(A):
    scaler = StandardScaler(with_mean=False)
    scaled_A = scaler.fit_transform(A)  # Assuming A is a CSR or CSC matrix
    corr_matrix = (1/scaled_A.shape[0]) * (scaled_A.T @ scaled_A)
    return corr_matrix

n_keyword = 10
DC_ttm_cor = {}
DF_TKW_cor = pd.DataFrame(columns=['Topic','Word','Correlate','Value'])
for topic_number in range(ldamodel_lda.n_topics):
    dtm_norm = DTM.copy()
    dtm_norm.data = dtm_norm.data / np.repeat(np.add.reduceat(dtm_norm.data, dtm_norm.indptr[:-1]), np.diff(dtm_norm.indptr))
    dtm_weigh = dtm_norm.transpose().multiply(csr_matrix(ldamodel_lda.doc_topic_.T[topic_number])).transpose()
    DC_ttm_cor[topic_number] = compute_sparse_correlation_matrix(dtm_weigh)
    for keyword in np.array(vocab)[np.argsort(ldamodel_lda.components_[topic_number])[::-1][:n_keyword]]:
        correlates = np.array(vocab)[np.argsort(DC_ttm_cor[topic_number][vocab==keyword].toarray()[0])[::-1][1:n_keyword]]
        values = DC_ttm_cor[topic_number][vocab==keyword].toarray()[0][np.argsort(DC_ttm_cor[topic_number][vocab==keyword].toarray()[0])[::-1][1:n_keyword]]
        DF_TKW_cor.loc[DF_TKW_cor.shape[0]]=[topic_number,keyword,correlates,values]

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
#authors_dict = dict(zip(authors,[authors_list.count(x) for x in authors]))
        
#DF_statistique_generale['nb_authors'] = DF_statistique_generale.Author.apply(lambda x: len(x))
        
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
# ########################################################### Author clustering
#==============================================================================

#TSNE (T-Distributed Stochastic Neighbouring Entities)
from sklearn.manifold import TSNE

tsne_model = TSNE(n_components=2, 
                  random_state=1234, 
                  init='pca',
                  perplexity=20.0,
                  early_exaggeration=4.0, 
                  learning_rate=10)

tsne_result = tsne_model.fit_transform(np.array(DF_AT[DF_AT.columns[:25]]))
DF_AT['x'] = tsne_result[:len(tsne_result),0]
DF_AT['y'] = tsne_result[:len(tsne_result),1]


from sklearn.cluster import KMeans
import datetime

k_range = 25 #range of k for kmean
k_numbers = list(range(k_range+1)[5:]) # Considering 1 to k, adding 1 as the last is cut off

kmean_models = {}
time_start = datetime.datetime.now()
for i in k_numbers:
    kmean_models[i] = KMeans(n_clusters=i, n_init = 100, random_state=1234).fit(DF_AT[topicnames].div(DF_AT.Pub_weighted,axis=0))
    sys.stdout.write("\rModeling till "+str(i)+" k took %s"%(str(datetime.datetime.now() - time_start)))
    sys.stdout.flush() 
    
for i in k_numbers:
    DF_AT[('Kmean_'+str(i))] = kmean_models[i].labels_

DF_CT = pd.DataFrame(columns=topicnames+['Cluster','Kmean'])   
for i in k_numbers:
    DF_temp = DF_AT[topicnames].groupby(kmean_models[i].labels_).sum()
    DF_temp = DF_temp.div(DF_temp.sum(axis=1),axis=0) #mean
    DF_temp['Cluster'] = list(range(len(DF_temp)))
    DF_temp['Kmean'] = 'Kmean_'+str(i)
    DF_CT = pd.concat([DF_CT,DF_temp],
                      axis=0,
                      join="inner",
                      ignore_index=True)
    
DF_CT_center = pd.DataFrame(columns=topicnames+['Cluster','Kmean'])   
for i in k_numbers:
    DF_temp = pd.DataFrame(kmean_models[i].cluster_centers_, columns=topicnames)
    DF_temp['Cluster'] = list(range(len(DF_temp)))
    DF_temp['Kmean'] = 'Kmean_'+str(i)
    DF_CT_center = pd.concat([DF_CT_center,DF_temp],
                             axis=0,
                             join="inner",
                             ignore_index=True)

#==============================================================================
# ########################################################### Topic correlation
#==============================================================================

# Topic Pearson Correlation
DF_TfromD = df_document_topic.corr(method='pearson')

DF_TfromW=df_topic_keywords.T.corr(method='pearson')

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

DIC_DF_ATP_corr_weight_filtered = dict()
for idx,period_name in enumerate(sorted(set(DF_statistique_generale.Period))):
    print(idx, period_name)
    list_index = [i for i in DIC_DF_ATP_corr[period_name].index if i in (DF_AT['Pub_w_'+str(idx+1)] > 1)]
    DIC_DF_ATP_corr_weight_filtered[period_name] = DIC_DF_ATP[period_name].loc[list_index].astype('float64').T.corr(method='pearson')
    
DIC_DF_ATP_corr_weight_filtered_high = dict()
for period_name in set(DF_statistique_generale.Period):
    DIC_DF_ATP_corr_weight_filtered_high[period_name] = pd.DataFrame(np.where(DIC_DF_ATP_corr_weight_filtered[period_name]<=.5, 0,DIC_DF_ATP_corr_weight_filtered[period_name]),
                 index=DIC_DF_ATP_corr_weight_filtered[period_name].index,
                 columns=DIC_DF_ATP_corr_weight_filtered[period_name].columns)
    
#save to csv
file_path = os.path.join(main_path,
                         "Diachronic and author analysis",
                         "Authors_corr",
                         "DF_Authors_corr_ALL_")
DF_AfromT.to_csv(file_path+"plain.csv",encoding="utf-8")

#### added later ...
#DF_AfromT.where(lambda x: x >= 0, '').to_csv(file_path+"plain_0_more.csv",encoding="utf-8")
#DF_AfromT.where(lambda x: x >= 0.01, '').to_csv(file_path+"plain_001_more.csv",encoding="utf-8")
#DF_AfromT.where(lambda x: x >= 0.1, '').to_csv(file_path+"plain_01_more.csv",encoding="utf-8")



DF_AfromT_Pub_weighted_1.to_csv(file_path+"weight.csv",encoding="utf-8")
DF_AfromT_Pub_weighted_2.to_csv(file_path+"highest.csv",encoding="utf-8")

for period_name in set(DF_statistique_generale.Period):
    file_path = os.path.join(main_path,
                             "Diachronic and author analysis",
                             "Authors_corr",
                             "DF_Authors_corr_"+period_name)
    DIC_DF_ATP_corr[period_name].to_csv(file_path+"_plain.csv",encoding="utf-8")
    DIC_DF_ATP_corr_weight_filtered[period_name].to_csv(file_path+"_weight.csv",encoding="utf-8")
    DIC_DF_ATP_corr_weight_filtered_high[period_name].to_csv(file_path+"_highest.csv",encoding="utf-8")

#### added later ...    
#period_name = '1996-2017'
#file_path = os.path.join(main_path,
#                             "Diachronic and author analysis",
#                             "Authors_corr",
#                             "DF_Authors_corr_"+period_name)  
#DIC_DF_ATP_corr[period_name].where(lambda x: x >= 0, '').to_csv(file_path+"_plain_0_more.csv",encoding="utf-8")
#DIC_DF_ATP_corr[period_name].where(lambda x: x >= 0.01, '').to_csv(file_path+"_plain_001_more.csv",encoding="utf-8")
#DIC_DF_ATP_corr[period_name].where(lambda x: x >= 0.1, '').to_csv(file_path+"_plain_01_more.csv",encoding="utf-8")

### added later
file_path = os.path.join(main_path,
                         "Diachronic and author analysis",
                         "Authors_corr")

DF_all = pd.read_csv(os.path.join(file_path,"DF_Authors_corr_ALL_plain.csv"),
                     sep=",",encoding="utf-8",header=0,index_col=0)

DF_all.where(lambda x: x >= 0, '').to_csv(os.path.join(file_path,"DF_Authors_corr_plain_0_more.csv"),encoding="utf-8")
DF_all.where(lambda x: x >= 0.01, '').to_csv(os.path.join(file_path,"DF_Authors_corr_plain_001_more.csv"),encoding="utf-8")
DF_all.where(lambda x: x >= 0.1, '').to_csv(os.path.join(file_path,"DF_Authors_corr_plain_01_more.csv"),encoding="utf-8")

DF_96 = pd.read_csv(os.path.join(file_path,"DF_Authors_corr_1996-2017_plain.csv"),sep=",",encoding="utf-8",header=0,index_col=0)

DF_96.where(lambda x: x >= 0, '').to_csv(os.path.join(file_path,"DF_Authors_corr_1996-2017_plain_0_more.csv"),encoding="utf-8")
DF_96.where(lambda x: x >= 0.01, '').to_csv(os.path.join(file_path,"DF_Authors_corr_1996-2017_plain_001_more.csv"),encoding="utf-8")
DF_96.where(lambda x: x >= 0.1, '').to_csv(os.path.join(file_path,"DF_Authors_corr_1996-2017_plain_01_more.csv"),encoding="utf-8")

### some df manipulations

DIC_DF_A = {'1930-1951' : pd.DataFrame(data='', index=range(len(authors)),columns=topicnames+['Author','Pub_sum','Pub_weighted']),
            '1952-1973' : pd.DataFrame(data='', index=range(len(authors)),columns=topicnames+['Author','Pub_sum','Pub_weighted']),
            '1974-1995' : pd.DataFrame(data='', index=range(len(authors)),columns=topicnames+['Author','Pub_sum','Pub_weighted']),
            '1996-2017' : pd.DataFrame(data='', index=range(len(authors)),columns=topicnames+['Author','Pub_sum','Pub_weighted'])}

for idx,author in enumerate(authors):
    list_bool = DF_statistique_generale.Author.apply(lambda x: True if author in x else False)
    for period in ['1930-1951','1952-1973','1974-1995','1996-2017']:
        list_bool = DF_statistique_generale.Author.apply(lambda x: True if author in x else False) & DF_statistique_generale.Period.isin([period])
        if max(list_bool):
            author_topic=sum(lda_output[list_bool]/np.transpose(np.repeat(np.expand_dims(DF_statistique_generale.nb_authors[list_bool],axis=0), repeats=25, axis=0)))
            #author_topic_norm=author_topic/sum(author_topic)
            DIC_DF_A[period].loc[idx]=list(author_topic)+[author,
                  authors_list.count(author),
                  sum(1/DF_statistique_generale.nb_authors[list_bool])]
            
DF_color = pd.read_csv(os.path.join(main_path,
                                      "Diachronic and author analysis",
                                      "Authors_period",
                                      "Topic_ID_colors.csv"),
    sep=";",encoding="utf-8",header=0,index_col=None)

for period in ['1930-1951','1952-1973','1974-1995','1996-2017']:
    file_path = os.path.join(main_path,
                             "Diachronic and author analysis",
                             "Authors_period",
                             "DF_Authors_NODES_modularity_"+period)
    DF_authors = pd.read_csv(file_path+".csv",sep=",",encoding="utf-8",header=0,index_col=None)
    DF_authors[['Pub_sum','Pub_weighted','Topic_ID_A','Topic_color_A','Category_color_A','Topic_name_A','Topic_ID_C','Topic_color_C','Category_color_C','Topic_name_C']+topicnames] = ''
    for i in range(len(DF_authors)):
        A_ID = DF_authors.Id[i]
        T_ID = np.argmax(DIC_DF_A[period].loc[A_ID][topicnames])
        for topic in topicnames:
            DF_authors[topic][i] = DIC_DF_A[period][topic][A_ID]
        DF_authors.Topic_ID_A[i] = T_ID
        DF_authors.Topic_color_A[i] = DF_color.Color[T_ID]
        DF_authors.Category_color_A[i] = DF_color.Color_category[T_ID]
        DF_authors.Topic_name_A[i] = DF_color.Topic_name[T_ID]
        DF_authors.Pub_sum[i] = DIC_DF_A[period].Pub_sum[A_ID]
        DF_authors.Pub_weighted[i] = DIC_DF_A[period].Pub_weighted[A_ID]    
    DF_authors.to_csv(os.path.join(file_path+"_v2.csv"),encoding="utf-8")
    
DIC_DF_C = {'1930-1951' : pd.DataFrame(columns=topicnames+['Topic_ID','Pub_sum','Pub_weighted']),
            '1952-1973' : pd.DataFrame(columns=topicnames+['Topic_ID','Pub_sum','Pub_weighted']),
            '1974-1995' : pd.DataFrame(columns=topicnames+['Topic_ID','Pub_sum','Pub_weighted']),
            '1996-2017' : pd.DataFrame(columns=topicnames+['Topic_ID','Pub_sum','Pub_weighted'])}

writer = pd.ExcelWriter(os.path.join(main_path,
                                         "Diachronic and author analysis",
                                         "Authors_period",
                                         "DF_Class.xlsx"))
for period in ['1930-1951','1952-1973','1974-1995','1996-2017']:
    file_path = os.path.join(main_path,
                             "Diachronic and author analysis",
                             "Authors_period",
                             "DF_Authors_NODES_modularity_"+period+"_v2.csv")
    DF_authors = pd.read_csv(file_path,sep=",",encoding="utf-8",header=0,index_col=None)
    modularity_class = set(DF_authors.modularity_class)
    for i in modularity_class:
        list_bool = DF_authors.modularity_class==i
        #class_topic=sum(np.array(DF_authors[topicnames][list_bool]))/sum(sum(np.array(DF_authors[topicnames][list_bool])))
        class_topic=sum(np.array(DF_authors[topicnames][list_bool])/np.transpose(np.repeat(np.expand_dims(DF_authors.Pub_weighted[list_bool],axis=0), repeats=25, axis=0)))
        DIC_DF_C[period].loc[i]=list(class_topic)+[np.argmax(class_topic),
                  sum(DF_authors.Pub_sum[list_bool]),
                  sum(DF_authors.Pub_weighted[list_bool])]    
    DIC_DF_C[period].to_excel(writer,period,encoding="utf-8")
writer.save()
writer.close()
        
from gensim.matutils import hellinger    

DF_all = pd.DataFrame(index=range(len(topicnames)))
for period in ['1930-1951','1952-1973','1974-1995','1996-2017']:
    for classe in range(len(DIC_DF_C[period])):
        DF_all[period+"_"+str(classe)] = np.array(DIC_DF_C[period].loc[classe][topicnames])

DF_hel = pd.DataFrame(index=DF_all.columns,columns=DF_all.columns)
for idx in DF_hel.index:
    for col in DF_hel.columns:
        dist = hellinger(DF_all[idx],DF_all[col])
        DF_hel[col][idx]=dist

writer = pd.ExcelWriter(os.path.join(main_path,
                                     "Diachronic and author analysis",
                                     "Authors_period",
                                     "DF_Class_Hellinger.xlsx"))
DF_hel.to_excel(writer,'Dist',encoding="utf-8")
writer.save()
writer.close()

for period in ['1930-1951','1952-1973','1974-1995','1996-2017']:
    file_path = os.path.join(main_path,
                             "Diachronic and author analysis",
                             "Authors_period",
                             "DF_Authors_NODES_modularity_"+period+"_v2.csv")
    DF_authors = pd.read_csv(file_path,sep=",",encoding="utf-8",header=0,index_col=0)
    for i in range(len(DF_authors)):
        C_ID = DF_authors.modularity_class[i]
        T_ID = DIC_DF_C[period].Topic_ID[C_ID]        
        DF_authors.Topic_ID_C[i] = T_ID
        DF_authors.Topic_color_C[i] = DF_color.Color[T_ID]
        DF_authors.Category_color_C[i] = DF_color.Color_category[T_ID]
        DF_authors.Topic_name_C[i] = DF_color.Topic_name[T_ID]           
    DF_authors.to_csv(os.path.join(file_path),encoding="utf-8")
    
#test p, p+1
DF_hel_test = pd.DataFrame(index=DIC_DF_C['1930-1951'].index,columns=DIC_DF_C['1952-1973'].index)
for class1 in range(len(DIC_DF_C['1930-1951'])):
    for class2 in range(len(DIC_DF_C['1952-1973'])):
        dist = hellinger(DIC_DF_C['1930-1951'].loc[class1][topicnames],DIC_DF_C['1952-1973'].loc[class2][topicnames])
        #dist = hellinger(DIC_DF_C['1930-1951'].loc[class1][topicnames]/sum(DIC_DF_C['1930-1951'].loc[class1][topicnames]),DIC_DF_C['1952-1973'].loc[class2][topicnames]/sum(DIC_DF_C['1930-1951'].loc[class1][topicnames]))
        DF_hel_test[class2][class1]=dist

writer = pd.ExcelWriter(os.path.join(main_path,
                                     "Diachronic and author analysis",
                                     "Authors_period",
                                     "DF_Class_Hellinger_test.xlsx"))
DF_hel_test.to_excel(writer,'test',encoding="utf-8")
writer.save()
writer.close()
  
#==============================================================================
# ################################################################ Save results
#==============================================================================
        
# Save lda results to excel
writer = pd.ExcelWriter(os.path.join(main_path,
                                     "Diachronic and author analysis",
                                     "Results_from_diachronic_and_author_analyses_v9.xlsx"))
df_param.T.to_excel(writer,'Para Score',encoding='utf8')    
pd.DataFrame([Vocab,[x[0] for x in DTM.T.sum(axis=1).tolist()]]).T.to_excel(writer,'Vocab',encoding='utf8')    
DF_topic.to_excel(writer,'Doc vs Topic',encoding='utf8')
DF_Topic_TKW.to_excel(writer,'Top 50 Topics Words',encoding='utf8')
DF_TKW_cor.to_excel(writer,'Top 10 Correlates',encoding='utf8')
df_keywords_topic.to_excel(writer,'Words vs Topics',encoding='utf8',
                           header=topicnames,
                           index=sorted([i for i in Vocab.keys()]))
DF_AT.to_excel(writer,'Authors vs Topics',encoding='utf8')
DF_CT.to_excel(writer,'Cluster mean vs Topics',encoding='utf8')
DF_CT_center.to_excel(writer,'Cluster center vs Topics',encoding='utf8')
DF_PAT.to_excel(writer,'Authors+P vs Topics',encoding='utf8')
DF_TP.to_excel(writer,'Topics vs Periods',encoding='utf8')
DF_TP_Overall.to_excel(writer,'Overall Topics vs Periods',encoding='utf8')
DF_PT_T10A.to_excel(writer,'Top 10 articles',encoding='utf8')
DF_PT_T20A.to_excel(writer,'Top 20 articles',encoding='utf8')
DF_TfromD.to_excel(writer,'Topic Cor. from Doc',encoding='utf8')
DF_TfromW.to_excel(writer,'Topic Cor. from Word',encoding='utf8')
DF_AfromT.to_excel(writer,'Author Cor.',encoding='utf8')
writer.save()
writer.close()
