import random

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def shuffle_two_arrays(a, b):
    combined = list(zip(a, b))
    random.shuffle(combined)
    a, b = zip(*combined)
    return a, b


class HardClustering:
    def __init__(self, texts, vectors, similarity_function='cosine_in_average', initial_threshold=0.9, step=0.05,
                 number_target_clusters=50, shuffle=True, random_state=41, strategy='one-to-one'):
        """

        :param vectors: вектора коллекции шуток
        :param similarity_function: коллекция шуток
        :param initial_threshold: начальный порог, должен быть максимально большим
        :param step: шаг, с которым после каждого просмотра коллекции уменьшается порог похожести, изначально равный
        initial_threshold
        :param number_target_clusters: целевое число кластеров
        :param shuffle: нужно ли перемешать коллекцию шуток и векторов
        :param random_state: фиксирование перестановок для воспроизводимости результатов
        """
        self.__texts = texts
        self.__vectors = vectors

        self.__sim_func = similarity_function
        self.__initial_threshold = initial_threshold
        self.__current_threshold = self.__initial_threshold
        self.__step = step

        self.__vector_clusters = [[]] * number_target_clusters
        self.__textual_clusters = [[]] * number_target_clusters

        self.__shuffle = shuffle
        self.__strategy = strategy

        self.__similarity_methods = {'cosine_in_average': self.__cosine_average_similarity}

        np.random.seed(random_state)

        if self.__shuffle:
            self.__texts, self.__vectors = shuffle_two_arrays(self.__texts, self.__vectors)

        self.__initial_size = len(self.__texts)

    def one_cycle(self, need_shuffle=True):
        """
        Делает один проход по коллекции.
        Один проход заключается в том, что: каждая шутка будет добавлена либо в существующий кластер, либо
        в пустой кластер (если пустые еще есть), либо будет возвращена в коллекцию. Попытка добавить происходит
        на основании текущего значчения похожести.
        :return: None
        """
        temp_vectors = []
        temp_texts = []
        sim_function = self.__similarity_methods.get(self.__sim_func)

        while self.__vectors:
            current_vector = self.__vectors.pop()  # удаление последнего элемента
            current_text = self.__texts.pop()

            insert, cluster_index = sim_function(current_vector)  # на данный момент шутку можно отнести только к
            # одному кластеру

            if not insert:
                found_free_index = self.__get_first_free_cluster_index()
                if not found_free_index:
                    temp_vectors.append(current_vector)
                    temp_texts.append(current_text)
                else:
                    self.__insert_in_cluster(found_free_index, current_vector, current_text)

            else:
                self.__insert_in_cluster(cluster_index, current_vector, current_text)

        self.__current_threshold -= self.__step

        if need_shuffle:
            self.__vectors, self.__texts = shuffle_two_arrays(self.__vectors, self.__texts)
        else:
            self.__vectors = temp_vectors
            self.__texts = temp_texts

    def display_current_cycle(self, show_top_n=5):
        """
        Отображает сводную статистику после запуска one_cycle, а также для каждого кластера выводит не более
        show_top_n шуток
        :return: None
        """
        if self.__initial_threshold == self.__current_threshold:
            raise ValueError('Кластера пустые, требуется вызов one_cycle')
        sizes = [len(x) for x in self.__vector_clusters]
        print('Median: {}, Mean: {}, Std: {}'.format(np.median(sizes), np.mean(sizes), np.std(sizes)))
        print('Всего кластеризованао {} из {}, текущий порог похожести: {}'.format(sum(sizes), self.__initial_size,
                                                                                   self.__current_threshold))
        print('Сэмплирование не более чем {} шуток из каждого кластера:')
        for index, cluster in enumerate(self.__textual_clusters):
            print('Cluster {}:\n{}\n'.format(index, '\n'.join(cluster[0:min(show_top_n, len(cluster) - 1)])))

    def reset_clusters(self, search_for_target_clusters=50, new_current_similarity=None, new_step=None):
        """
        Обновление кластеров (стирание текущего построения)
        :param search_for_target_clusters: новое целевое количество кластеров
        :param new_current_similarity: обновление начальной похожести
        :param new_step: обновление уменшения похожести
        :return: None
        """
        self.__vector_clusters = [[]] * search_for_target_clusters
        self.__textual_clusters = [[]] * search_for_target_clusters

        if new_current_similarity:
            self.__current_threshold = new_current_similarity

        if new_step:
            self.__step = new_step

    def get_current_clusters(self):
        """
        Возвращает текущее построение, вектора и сами шутки
        :return:
        """
        return self.__vector_clusters, self.__textual_clusters

    def __get_first_free_cluster_index(self):
        for i in range(len(self.__vector_clusters)):
            if not self.__vector_clusters[i]:
                return i
        return None

    def __insert_in_cluster(self, cluster_index, vector, text):
        self.__vector_clusters[cluster_index].append(vector)
        self.__textual_clusters[cluster_index].append(text)

    def __cosine_average_similarity(self, vector):
        similarities = [np.mean(cosine_similarity(vector, cluster)[0]) if cluster else 0.0
                        for cluster in self.__vector_clusters]
        max_sim_index = np.argmax(similarities)
        max_sim = similarities[max_sim_index]
        if max_sim < self.__current_threshold:
            return False, None
        return True, max_sim_index

        # TODO вектора могут относиться к нескольким кластерам: np.argwhere(listy == np.amax(listy))


if __name__ == '__main__':
    pass
