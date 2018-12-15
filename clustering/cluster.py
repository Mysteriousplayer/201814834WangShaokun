import json
from sklearn.cluster import *
from sklearn import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import cluster, datasets, mixture, preprocessing
texts = []
labels = []
input = open('C:\\Users\\Administrator\\Desktop\\Tweets.txt', 'r')
for line in input.readlines():
    tweets = json.loads(line)
    texts.append(tweets['text'])
    labels.append(tweets['cluster'])

vectorizer = TfidfVectorizer()
vec = vectorizer.fit_transform(texts)
vectorizer_2 = CountVectorizer()
vec_w2v = vectorizer_2.fit_transform(texts)

# KMeans
clf = KMeans(n_clusters=100)
a = clf.fit(vec)
labels_predict = clf.labels_
nml = normalized_mutual_info_score(labels, labels_predict)
print('the nml of Kmeans:', nml)

# Affinity Propagation
afp = AffinityPropagation().fit(vec)
cluster_centers_indices = afp.cluster_centers_indices_
labels_predict = afp.labels_
nml = normalized_mutual_info_score(labels, labels_predict)
print('the nml of Affinity Propagation:', nml)

# MeanShift
vec_w2v_a = preprocessing.scale(vec_w2v.toarray())
clustering = MeanShift(bandwidth=5).fit(vec_w2v_a)
labels_predict = clustering.labels_
nml = normalized_mutual_info_score(labels, labels_predict)
print('the nml of MeanShift:', nml)

# SpectralClustering
spec = SpectralClustering(n_clusters=100)
spec.fit(vec)
labels_predict = spec.labels_
nml = normalized_mutual_info_score(labels, labels_predict)
print('the nml of Spectral Clustering:', nml)

# Ward Hierarchical Clustering
wh = AgglomerativeClustering(n_clusters=100).fit(vec.toarray())
labels_predict = wh.labels_
nml = normalized_mutual_info_score(labels, labels_predict)
print('the nml of Ward Hierarchical Clustering:', nml)

# Agglomerative Clustering
agg = AgglomerativeClustering(linkage='complete', n_clusters=100).fit(vec.toarray())
labels_predict = agg.labels_
nml = normalized_mutual_info_score(labels, labels_predict)
print('the nml of Agglomerative Clustering:', nml)

# DBSCAN
db = DBSCAN(eps=0.3, min_samples=1).fit(vec_w2v.todense())
labels_predict = db.labels_
nml = normalized_mutual_info_score(labels, labels_predict)
print('the nml of DBSCAN:', nml)

# Gaussian Mixtures
gmm = mixture.GaussianMixture(n_components=100, covariance_type='diag')
gmm.fit(vec.toarray())
labels_predict = gmm.predict(vec.toarray())
nml = normalized_mutual_info_score(labels, labels_predict)
print('the nml of Gaussian Mixtures:', nml)