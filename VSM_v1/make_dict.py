import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os
import sys
from gensim import corpora ,models, similarities
from collections import defaultdict
from pprint import  pprint              ##pretty-printer
from six import iteritems
import logging                          ## 引入日志配置
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#for i in range(0,10):
doc=[]
rootdir = 'C:\\Users\\Administrator\\Desktop\\20news-18828'
list1 = os.listdir(rootdir) #列出文件夹下所有的目录与文件
for i in range(0,len(list1)):
    path1 = os.path.join(rootdir,list1[i])
    #fname="C:\\Users\\Administrator\\Desktop\\VSM\\data\\"+str(82758+i)
    #print(path1)
    list2 = os.listdir(path1)
    for j in range(0,len(list2)):
        path2 = os.path.join(path1,list2[j])
        with open(path2,'r',encoding="utf-8") as f:
        
            #print(path2)
            #print(f.read().encode('utf-8').decode('utf-8-sig'))
            doc.append(f.read().encode('utf-8').decode('utf-8-sig'))
print(len(doc))
stoplist = set('for a of the and to in \n * = > < + ^ # ! @ & $ ~'.split())     ##停留词
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in doc]
frequency = defaultdict(int)                         ##设置默认词频
for text in texts:
    for token in text:
        frequency[token] += 1
print(len(frequency))
texts = [[token for token in text if frequency[token] >30]   ##删除仅仅出现一次的词
         for text in texts]
#pprint(texts)
dictionary = corpora.Dictionary(texts)
dictionary.save('C:\\Users\\Administrator\\Desktop\\VSM\\test.dict')
#print(dictionary.token2id)