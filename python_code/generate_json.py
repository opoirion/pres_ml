# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


training_set = [
    "super ce blog! J'adore ce truc...",
    "De la balle! c'est vraiment super!",
    "que des bonnes choses, bien fait et très intéressant",
    "pas terrible c'est vraiment un blog de gros naze...",
    "On se fout de ma geule! remboursez!!! c'est naze!",
    "pas super ce blog, peut mieux faire je n'y reviendrai pas",
    "c'est un super commentaire positif",
    "pas super",
]

groups = [
    0,0,0,
    1,1,1,
    2,2
]


def main():
    vectorizer = CountVectorizer(ngram_range=(1,2),max_df=1.0, min_df=0.0)

    nei = NearestNeighbors(algorithm='brute', metric='jaccard')
    matrix = vectorizer.fit_transform(training_set)
    nei.fit(matrix.todense())
    import ipdb;ipdb.set_trace()
    
if __name__ == "__main__":
    main()
