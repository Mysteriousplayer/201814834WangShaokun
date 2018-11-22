import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os
import sys
from gensim import corpora ,models, similarities
from collections import defaultdict
from pprint import  pprint              ##pretty-printer
from six import iteritems
token_sum=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
file_sum=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
true_class=[]
predict_class=[]
doc_class=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
rootdir = 'C:\\Users\\Administrator\\Desktop\\20news-18828'
list1 = os.listdir(rootdir) #列出文件夹下所有的目录与文件

for i in range(0,len(list1)):
    path1 = os.path.join(rootdir,list1[i])
    #fname="C:\\Users\\Administrator\\Desktop\\VSM\\data\\"+str(82758+i)
    #print(path1)
    list2 = os.listdir(path1)
    doc=[]
    for j in range(0,len(list2)):
        true_class.append(i)
        path2 = os.path.join(path1,list2[j])
        with open(path2,'r',encoding="utf-8") as f:
            #print(path2)
            #print(f.read().encode('utf-8').decode('utf-8-sig'))
            doc.append(f.read().encode('utf-8').decode('utf-8-sig'))
    #print(len(doc))
    file_sum[i]=len(doc)
    stoplist = set('for a of the and to in \n * = > < + ^ # ! @ & $ ~ ? == ** ( ) << >> - _'.split())     ##停留词
    texts = [[word for word in document.lower().split() if word not in stoplist]
            for document in doc]
    frequency = defaultdict(float)                       ##设置默认词频
    for text in texts:
        for token in text:
            frequency[token] += 1
    #print(len(frequency))
    texts = [[token for token in text if frequency[token] >10]   ##删除仅仅出现一次的词
            for text in texts]
    frequency_touse = defaultdict(float)
    for text in texts:
        for token in text:
            frequency_touse[token] += 1
            token_sum[i]+=1
    doc_class[i]=frequency_touse
    #print(frequency_touse)
print(token_sum)
print(file_sum)
#print(true_class)
allfile=0
index=0
for l in range(0,len(file_sum)):
    allfile+=file_sum[l]

for i in range(0,len(list1)):
    path1 = os.path.join(rootdir,list1[i])
    list2 = os.listdir(path1)
    
    for j in range(0,len(list2)):
        #true_class.append(i)
        path2 = os.path.join(path1,list2[j])
        with open(path2,'r',encoding="utf-8") as f:
            document=f.read().encode('utf-8').decode('utf-8-sig')
        stoplist = set('for a of the and to in \n * = > < + ^ # ! @ & $ ~ ? == ** ( ) << >> - _'.split())     ##停留词
        text = [word for word in document.lower().split() if word not in stoplist]
        likelihood=[100000000000.0,100000000000.0,100000000000.0,100000000000.0,100000000000.0,100000000000.0,100000000000.0,100000000000.0,100000000000.0,100000000000.0,100000000000.0,100000000000.0,100000000000.0,100000000000.0,100000000000.0,100000000000.0,100000000000.0,100000000000.0,100000000000.0,100000000000.0]
        #print(likelihood)
        not_in=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for k in range(0,len(list1)):
            a=float(file_sum[k]/allfile)
            likelihood[k]*=a
        #print(likelihood)
        for token in text:
            for k in range(0,len(list1)):
                if doc_class[k].__contains__(token)==False:
                    not_in[k]+=1
        for token in text:
            for k in range(0,len(list1)):
                if doc_class[k].__contains__(token)==False:
                    b=float(1.0/(token_sum[k]+not_in[k]))
                    likelihood[k]*=(b*10000.0)
                    #print('b')
                    #print(b)
                    #print(likelihood)
                if doc_class[k].__contains__(token)==True:
                    c=float(doc_class[k][token]/(token_sum[k]+not_in[k]))
                    likelihood[k]*=(c*10000.0)
                    #print('c')
                    #print(c)
                    #print(likelihood)
        #print(likelihood)
        prediction=likelihood.index(max(likelihood))
        #print(prediction)
        predict_class.append(prediction)    
        
for i in range(0,allfile):
    if predict_class[i]==true_class[i]:
        index+=1
accuary=index/allfile
print("the accuary of NBC:")
print(accuary)
            
            
            