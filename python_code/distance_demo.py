# -*- coding: utf-8 -*-

from scipy import random
from scipy.sparse import hstack

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from os.path import abspath
from os.path import split as pathsplit

from scipy import vstack

import json

dim = (50, 50)

name = "distance2"

matrix = vstack(
    (random.normal(size=(dim[0], 5), loc=-1),
     random.normal(size=(dim[1], 5), loc=1)
 )
)

groups = [0 for i in range(dim[0])] + [1 for i in range(dim[1])]

n_neighbors = 3

def main():

    nei = NearestNeighbors(metric='euclidean')
    nei.fit(matrix)
    path =  '{0}/'.format(pathsplit(abspath(__file__))[0])

    jsonfile = open(path + '{1}_rand-{0}-nn.json'.format(n_neighbors, name), 'w')

    nodes = [{'name': i,
              'group':groups[i]}
             for i in range(len(matrix))]
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
