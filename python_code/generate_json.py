# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from os.path import abspath
from os.path import split as pathsplit

import json

training_set = [
    "super ce blog! J'adore ce truc...",
    "De la balle! c'est vraiment super!",
    "que des bonnes choses, bien fait et très intéressant",
    "pas terrible c'est vraiment un blog de gros naze...",
    "On se fout de ma geule! remboursez!!! c'est naze!",
    "pas super ce blog, peut mieux faire je n'y reviendrai pas"
]

new_comments = [
    "c'est un super commentaire positif",
    "pas super",
]


groups = [
    0,0,0,
    10,10,10
]

new_groups = [
    20,30
]

n_neighbors = 3

def main():
    vectorizer = CountVectorizer(ngram_range=(1,2),max_df=1.0, min_df=0.0)

    nei = NearestNeighbors(algorithm='brute', metric='jaccard')
    matrix = vectorizer.fit_transform(training_set).todense()
    new_matrix = vectorizer.transform(new_comments).todense()
    nei.fit(matrix)
    path =  '{0}/'.format(pathsplit(abspath(__file__))[0])
    jsonfile = open(path + '{0}-nn.json'.format(n_neighbors), 'w')

    nodes = [{'name': (training_set+new_comments)[i],
              'group':(groups + new_groups)[i]}
             for i in range(len(training_set))]
    links = []
    
    for i in range(len(matrix)):
        dist, idnei = nei.kneighbors(matrix[i], n_neighbors=n_neighbors + 1)
        dist, idnei = dist[0], idnei[0]

        for j in range(len(idnei[1:])):
            links.append({"source":i,"target":idnei[j+1],"value":10*(1 - dist[j+1])})
    jsondumped = json.dumps({'nodes':nodes, 'links':links}, indent=2)

    jsonfile.write(jsondumped)    
if __name__ == "__main__":
    main()
