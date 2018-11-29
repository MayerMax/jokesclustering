from typing import List

import numpy
from gensim.utils import simple_preprocess

from vector_clustering.abstract.abstract_vectorizer import AbstractVectorizer
from vector_clustering.data.manager import load


class SimpleNormVectorizer(AbstractVectorizer):
    def __init__(self, embedding_mode=None):
        if embedding_mode is not None:
            self.__vectorizer = load(embedding_mode)
        else:
            self.__vectorizer = None
        super().__init__()

    def fit_text(self, text: str):
        try:
            vectors = [self.__vectorizer.wv[x] for x in simple_preprocess(text, max_len=1000000)
                       if x in self.__vectorizer.wv]
            return sum(vectors) / len(vectors)
        except ZeroDivisionError:
            print(text)

    def fit(self, texts: List[str]) -> numpy.ndarray:
        return numpy.array([self.fit_text(x) for x in texts])

    def custom_init(self, **kwargs):
        pass
