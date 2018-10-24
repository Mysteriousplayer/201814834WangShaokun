import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os
import sys
from gensim import corpora ,models, similarities
from collections import defaultdict
from pprint import  pprint              ##pretty-printer
from six import iteritems
import heapq

dict=corpora.Dictionary.load('C:\\Users\\Administrator\\Desktop\\VSM\\test.dict')
#print(dict.token2id)
#test_word="i love you so much"
#new_vec = dict.doc2bow(test_word.lower().split())
#print(new_vec)

doc=[]
doc_test=[]
rootdir = 'C:\\Users\\Administrator\\Desktop\\20news-18828'
list1 = os.listdir(rootdir) #列出文件夹下所有的目录与文件
l_index=[]#标出每个训练集数据的类别
l_index_2=[]#标出每个测试集数据的类别
all_prediction=[]
index_=0
for i in range(0,len(list1)):
    path1 = os.path.join(rootdir,list1[i])
    #fname="C:\\Users\\Administrator\\Desktop\\VSM\\data\\"+str(82758+i)
    #print(path1)
    list2 = os.listdir(path1)
    for j in range(0,int(0.8*len(list2))):
        l_index.append(index_)
        path2 = os.path.join(path1,list2[j])
        with open(path2,'r',encoding="utf-8") as f:
            #print(path2)
            #print(f.read().encode('utf-8').decode('utf-8-sig'))
            doc.append(f.read().encode('utf-8').decode('utf-8-sig'))
            
    for k in range(int(0.8*len(list2))+1,len(list2)):
        l_index_2.append(index_)
        all_prediction.append(0)
        path2 = os.path.join(path1,list2[k])
        with open(path2,'r',encoding="utf-8") as f:
            doc_test.append(f.read().encode('utf-8').decode('utf-8-sig'))
    index_=index_+1
print(len(doc))
print(len(doc_test))
stoplist = set('for a of the and to in \n * = > < + ^ # ! @ & $ ~'.split())     ##停留词
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in doc]
texts_test = [[word for word in document.lower().split() if word not in stoplist]
         for document in doc_test]

corpus_model= [dict.doc2bow(text) for text in texts]
tfidf_model = models.TfidfModel(corpus_model)
corpus_tfidf = tfidf_model[corpus_model]

#for i in range(0,1):
test_corpus_model=[dict.doc2bow(text) for text in texts_test]
index = similarities.MatrixSimilarity(corpus_tfidf) #把所有评论做成索引
K=25#KNN的参数
for i in range(len(doc_test)):
    new_tfidf=tfidf_model[test_corpus_model[i]]
    #print(new_tfidf)
    #index = similarities.MatrixSimilarity(corpus_tfidf) #把所有评论做成索引
    sims = index[new_tfidf]  #利用索引计算每一条评论和商品描述之间的相似度
    #print(sims)
    #print(sims[1])
    #print(l_index_2)
    sims_list=sims.tolist()
    results = map(sims_list.index, heapq.nlargest(K, sims_list))
    #results.sort()
    #print(result for result in results)
    label=[]
    for j in range(20):
        label.append(0)
    #print(label)
    for result in results:
        label[l_index[result]]+=1
    new_labels=map(label.index, heapq.nlargest(1, label))
    #print(label)
    prediction=label.index(max(label))
    print(prediction)
    all_prediction[i]=prediction

right_label=0
for i in range(len(doc_test)):
    if all_prediction[i]==l_index_2[i]:
        right_label+=1

accuary=right_label/len(doc_test)
print("the accuary of KNN(K="+str(K)+"):")
print(accuary)

