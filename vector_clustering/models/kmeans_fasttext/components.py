from typing import List, Union, Dict

import numpy
from sklearn.cluster import KMeans

from vector_clustering.abstract.abstract_cluster_extractor import AbstractClusterExtractor
from vector_clustering.abstract.abstract_vectorizer import AbstractVectorizer
from vector_clustering.abstract.model import Model
from vector_clustering.data.manager import load, ARANEUM_FASTTEXT


class KmeansVectorizer(AbstractVectorizer):
    def __init__(self):
        super().__init__()
        self.__fasttext = load(ARANEUM_FASTTEXT)

    def fit(self, texts: List[str]) -> numpy.ndarray:
        return numpy.array([self.__fasttext[text] for text in texts])

    def custom_init(self, **kwargs):
        pass


class KmeansSimpleClusterExtractor(AbstractClusterExtractor):
    def __init__(self, model:Union[None, KMeans] = None):
        super().__init__()
        if model:
            self.__model = model
        else:
            self.__model = KMeans(n_clusters=2)

    def fit(self, vectors: numpy.ndarray):
        self.__model.fit(vectors)

    def demo(self, n_examples, texts=None) -> Dict[Union[str, int], List[str]]:
        labels = self.__model.labels_
        demo_set = {label_id : [] for label_id in range(self.__model.cluster_centers_.shape[0])}
        for label_id in demo_set:
            satisfy_texts = texts[labels == label_id]
            demo_set[label_id] = satisfy_texts[0:min(n_examples, satisfy_texts.shape[0])]
        return demo_set

    def get_model(self):
        return self.__model


if __name__ == '__main__':
    texts = numpy.array(['my name is max', 'your name is max'])
    vectorizer = KmeansVectorizer()
    cluster_extractor = KmeansSimpleClusterExtractor()

    m = Model(vectorizer, cluster_extractor)
    print(m.fit(texts))