# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import jaccard

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
    "c'est un super commentaire positif",
    "pas super"
]

def main():
    vectorizer = CountVectorizer(ngram_range=(1,2),max_df=1.0, min_df=0.0)

    matrix = vectorizer.fit_transform(training_set)
    for new_comment in new_comments:
        print '\n####\ncommentaire:{0}\n'.format(new_comment)
        vector = vectorizer.transform([new_comment])

        i = 0
        for vect in matrix:
            score = jaccard(vect.todense(), vector.todense())
            print 'sentence: {0}"\tscore:{1}'.format(training_set[i], score)
            i += 1

if __name__ == "__main__":
    main()
