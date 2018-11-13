from abc import ABC, abstractmethod
from typing import List

import numpy


class AbstractVectorizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, texts: List[str]) -> numpy.ndarray:
        """
        Абстрактная функция, которая получает на вход набор текстов и возвращает их векторные представления
        :param texts: последовательность текстов
        :return: массив векторов текстов
        """
        pass

    @abstractmethod
    def custom_init(self, **kwargs):
        """
         Дополнительная функция для случаев, если нужно что-нибудь проинициализировать отдельно
        :param kwargs:
        :return:
        """
        pass