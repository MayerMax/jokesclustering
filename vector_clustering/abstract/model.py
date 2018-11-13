from typing import List

from vector_clustering.abstract.abstract_cluster_extractor import AbstractClusterExtractor
from vector_clustering.abstract.abstract_vectorizer import AbstractVectorizer

class Model:
    def __init__(self, vectorizer: AbstractVectorizer, cluster_extractor: AbstractClusterExtractor):
        self.__vectorizer = vectorizer
        self.__cluster_extractor = cluster_extractor

    def fit(self, texts: List[str]):
        vector_representation = self.__vectorizer.fit(texts)
        self.__cluster_extractor.fit(vector_representation)
        return self.__cluster_extractor.demo()