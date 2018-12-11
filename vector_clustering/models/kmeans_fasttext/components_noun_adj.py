from typing import List

import numpy
from gensim.utils import simple_preprocess

from vector_clustering.abstract.abstract_vectorizer import AbstractVectorizer
from vector_clustering.data.manager import load


class NounAdjVectorizer(AbstractVectorizer):
    def __init__(self, embedding_mode=None):
        if embedding_mode is not None:
            self.__vectorizer = load(embedding_mode)
        else:
            self.__vectorizer = None
        super().__init__()

    def fit_text(self, text: str):
        tokens = simple_preprocess(text)

        pass

    def fit(self, texts: List[str]) -> numpy.ndarray:
        return numpy.array([self.__vectorizer[text] for text in texts])

    def custom_init(self, **kwargs):
        pass
