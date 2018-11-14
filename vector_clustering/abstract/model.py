from typing import List, Dict, Union

import numpy

from vector_clustering.abstract.abstract_cluster_extractor import AbstractClusterExtractor
from vector_clustering.abstract.abstract_vectorizer import AbstractVectorizer


class Model:
    def __init__(self, vectorizer: AbstractVectorizer, cluster_extractor: AbstractClusterExtractor, n=10):
        self.__vectorizer = vectorizer
        self.__cluster_extractor = cluster_extractor
        self.__n = 10

    def fit(self, texts: numpy.ndarray) -> Dict[Union[str, int], List]:
        vector_representation = self.__vectorizer.fit(texts)
        print('vectorizer is fitted')
        self.__cluster_extractor.fit(vector_representation)
        print('clustering model is fitted')
        return self.__cluster_extractor.demo(self.__n, texts)

    def get_model(self) -> object:
        """
        Возвращает объект, с помощью которого делается кластеризация, если такой имеется или None
        :return: object
        """
        return self.__cluster_extractor.get_model()
