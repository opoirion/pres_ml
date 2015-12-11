# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import hstack

training_set = [
    "super ce blog! J'adore ce truc...",
    "De la balle! c'est vraiment super!",
    "que des bonnes choses, bien fait et très intéressant",
    "pas terrible c'est vraiment un blog de gros naze...",
    "On se fout de ma geule! remboursez!!! c'est naze!",
    "pas super ce blog, peut mieux faire je n'y reviendrai pas"
]



status = [
    "good comment",
    "good comment",
    "good comment",
    "bad comment",
    "bad comment",
    "bad comment"
]

new_comments = [
    "pas super ce commentaire!"
]

def main():
    vectorizer_ngram = TfidfVectorizer(ngram_range=(1,3))
    vectorizer_kmer = TfidfVectorizer(ngram_range=(1,3), analyzer='char')

    matrix_ngram = vectorizer_ngram.fit_transform(training_set)
    matrix_kmer = vectorizer_kmer.fit_transform(training_set)

    matrix = hstack((matrix_ngram, matrix_kmer))
    
    nei = KNeighborsClassifier(metric='cosine', algorithm='brute', n_neighbors=1)

    nei.fit(matrix.todense(), status)

    vector_kmer = vectorizer_kmer.transform(new_comments)
    vector_ngram = vectorizer_ngram.transform(new_comments)

    vector = hstack((vector_kmer, vector_ngram))
    print nei.predict(vector.todense())
    

if __name__ == "__main__":
    main()
