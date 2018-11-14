from abc import ABC, abstractmethod
from typing import List, Dict, Union

import numpy


class AbstractClusterExtractor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, vectors: numpy.ndarray):
        """
        Реализует выделение кластеров
        :param vectors: результат применения fit от abstract_vectorizer
        :return: None
        """
        pass

    @abstractmethod
    def demo(self, n_examples, texts=None) -> Dict[Union[str, int], List[str]]:
        """
        Функция демонстрации разбиения на кластеры.
        Возвращает для каждого кластера некоторое множество примеров (в исходном текстовом виде),
        объем фиксируется через n_examples.
        :return:
        """
        pass

    @abstractmethod
    def get_model(self) -> object:
        """
        Возвращает объект модели, с помощью которой будет производиться кластеризация, если такой имеется, либо None
        :return: object
        """
        pass