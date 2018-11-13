from abc import ABC, abstractmethod
from typing import List, Dict

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
    def demo(self) -> Dict[str, List[str]]:
        """
        Функция демонстрации разбиения на кластеры.
        Возвращает для каждого кластера некоторое множество примеров (в исходном текстовом виде)
        :return:
        """
        pass
