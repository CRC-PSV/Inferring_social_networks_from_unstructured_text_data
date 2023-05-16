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
from gensim.matutils import hellinger    

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

DF_color = pd.read_csv(os.path.join(main_path,
                                    "0. Data",
                                    "Topic_ID_colors.csv"),
                       sep=";",encoding="utf-8",header=0,index_col=None)

#==============================================================================
# #################################################### Author profile by period
#==============================================================================       

lda_output = ldamodel_lda.doc_topic_
topicnames = ["Topic_" + str(i) for i in range(len(ldamodel_lda.components_))]
authors = sorted(set([author for authors in DF_statistique_generale.Author for author in authors]))
authors_list = [author for authors in DF_statistique_generale.Author for author in authors]

DIC_DF_A = {'1930-1951' : pd.DataFrame(data='', index=range(len(authors)),columns=topicnames+['Author','Pub_sum','Pub_weighted']),
            '1952-1973' : pd.DataFrame(data='', index=range(len(authors)),columns=topicnames+['Author','Pub_sum','Pub_weighted']),
            '1974-1995' : pd.DataFrame(data='', index=range(len(authors)),columns=topicnames+['Author','Pub_sum','Pub_weighted']),
            '1996-2017' : pd.DataFrame(data='', index=range(len(authors)),columns=topicnames+['Author','Pub_sum','Pub_weighted'])}

for idx,author in enumerate(authors):
    list_bool = DF_statistique_generale.Author.apply(lambda x: True if author in x else False)
    for period in set(DF_statistique_generale.Period):
        list_bool = DF_statistique_generale.Author.apply(lambda x: True if author in x else False) & DF_statistique_generale.Period.isin([period])
        if max(list_bool):
            author_topic=sum(lda_output[list_bool]/np.transpose(np.repeat(np.expand_dims(DF_statistique_generale.nb_authors[list_bool],axis=0), repeats=25, axis=0)))
            DIC_DF_A[period].loc[idx]=list(author_topic)+[author,
                  authors_list.count(author),
                  sum(1/DF_statistique_generale.nb_authors[list_bool])]
            
#==============================================================================
# ####################################### Read Gephy output and add information
#==============================================================================

for period in set(DF_statistique_generale.Period):
    file_path = os.path.join(main_path,
                             "0. Data",
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
    
#==============================================================================
# ############################################################## HCol by period
#==============================================================================
    
DIC_DF_C = {'1930-1951' : pd.DataFrame(columns=topicnames+['Topic_ID','Pub_sum','Pub_weighted']),
            '1952-1973' : pd.DataFrame(columns=topicnames+['Topic_ID','Pub_sum','Pub_weighted']),
            '1974-1995' : pd.DataFrame(columns=topicnames+['Topic_ID','Pub_sum','Pub_weighted']),
            '1996-2017' : pd.DataFrame(columns=topicnames+['Topic_ID','Pub_sum','Pub_weighted'])}

writer = pd.ExcelWriter(os.path.join(main_path,
                                     "2. Data analysis and HCols identification",
                                     "2.4 HCols distances",
                                     "DF_Class.xlsx"))
for period in set(DF_statistique_generale.Period):
    file_path = os.path.join(main_path,
                             "Diachronic and author analysis",
                             "Authors_period",
                             "DF_Authors_NODES_modularity_"+period+"_v2.csv")
    DF_authors = pd.read_csv(file_path,sep=",",encoding="utf-8",header=0,index_col=None)
    modularity_class = set(DF_authors.modularity_class)
    for i in modularity_class:
        list_bool = DF_authors.modularity_class==i
        class_topic=sum(np.array(DF_authors[topicnames][list_bool])/np.transpose(np.repeat(np.expand_dims(DF_authors.Pub_weighted[list_bool],axis=0), repeats=25, axis=0)))
        DIC_DF_C[period].loc[i]=list(class_topic)+[np.argmax(class_topic),
                  sum(DF_authors.Pub_sum[list_bool]),
                  sum(DF_authors.Pub_weighted[list_bool])]    
    DIC_DF_C[period].to_excel(writer,period,encoding="utf-8")
writer.save()
writer.close()
        
#==============================================================================
# ########################################################## Hellinger distance
#==============================================================================    

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
                                     "2. Data analysis and HCols identification",
                                     "2.4 HCols distances",
                                     "DF_Class_Hellinger.xlsx"))
DF_hel.to_excel(writer,'Dist',encoding="utf-8")
writer.save()
writer.close()
