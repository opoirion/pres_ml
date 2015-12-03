# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer


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

def main():
    vectorizer = TfidfVectorizer(
        ngram_range=(1,1),
        use_idf=False,
        smooth_idf = False,
        norm=None)
    matrix = vectorizer.fit_transform(training_set)
    print matrix

if __name__ == "__main__":
    main()
