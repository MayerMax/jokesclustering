import os

import pandas
from gensim.models import FastText

ARANEUM_FASTTEXT = 0


def load(constant: int):
    if constant not in __loading_mappers:
        raise ValueError('Неизвестный идентификатор модели')
    return __loading_mappers.get(constant)()


def get_jokes_as_dataframe():
    return pandas.read_json(os.path.join(os.path.dirname(__file__), 'jokes.json'), encoding='utf-8').rename(index=str,
                                                                                          columns={0: "joke_text"})


def load_pandas_csv(file_name):
    """
    файл лежит в директории data или в подпапке
    :param file_name: имя пути до файл относительно папки data
    :return: pandas
    """
    return pandas.read_csv(os.path.join(os.path.dirname(__file__), file_name))


def __load_araneum_fasttext():
    return FastText.load(os.path.join(os.path.dirname(__file__),
                                      'araneum_fasttext/araneum_none_fasttextcbow_300_5_2018.model'))


__loading_mappers = {
    ARANEUM_FASTTEXT: __load_araneum_fasttext
}
