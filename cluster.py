# unix file system libraries
import glob
import os

# scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


"""
  Options
"""
path = 'docs4'      # directory of articles .txt files
k = 5               # number of centroids i.e. clusters
t = 5               # number of terms in print out



"""
  Documents
    ...get documents from folder and create Document objects for each
"""
dataset = []

for name in os.listdir(path):
  with open(os.path.join(path, name), 'r') as f:
    datum = f.read().replace('\n', ' ')
    dataset.append(datum)

titles = [title for title in os.listdir(path)]



"""
  TF-IDF weighting
  KMeans clustering
"""
vectorizer = TfidfVectorizer(
  #max_df=0.5,
  max_features=None,                # default
  #min_df=1.0,                       # default
  stop_words='english',
  use_idf=True
  )

X = vectorizer.fit_transform(dataset)

km = KMeans(
  n_clusters=k,                     # default = 8, num centroids i.e. num clusters
  init='k-means++',
  max_iter=100,
  n_init=10,                        # default
  verbose=0                         # default
  )

km.fit(X)




"""
  Data processing
"""
uniqueLabels = list(set(km.labels_))    # list of unique labels

# dicts
clusters = {}           # cluster id : article titles
clusterTerms = {}       # cluster id : terms
for l in uniqueLabels:
  clusters[l] = []
  clusterTerms[l] = []

# arrange in dict according to label
for i in range(len(titles)):
  clusters[km.labels_[i]].append(titles[i])

# get terms
centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(k):
  for ind in centroids[i, :t]:
    # arrange the terms in the dict, according to label
    clusterTerms[i].append(terms[ind])

# print contents of each cluster i.e. articles in each one
for key in clusters:
  print 'Cluster {}:'.format(key+1)
  print 'Terms: {}'.format(clusterTerms[key])
  for value in clusters[key]:
    print '\t{}'.format(value)
  print
