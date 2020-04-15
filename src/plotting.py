from typing import List
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    sns.set_style('whitegrid')
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]

    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()


# Plotting Module
class Graph:
    def __init__(self):
        self.figure, self.graph = plt.subplots(1, 1)

    def set_labels(self, title: str = "Title", x: str = "X-axis", y: str = "Y-axis", fontsize=None):
        if fontsize is None or len(fontsize) < 3:
            print("Invalid fontsize for labels")
            fontsize = [20, 14, 14]
        self.graph.set_title(title, fontsize=fontsize[0])
        self.graph.set_xlabel(x, fontsize=fontsize[1])
        self.graph.set_ylabel(y, fontsize=fontsize[2])

    def plot(self, x: List = None, y: List = None, label=None):
        self.graph.plot(x, y, label=label)

    def legend(self, title, loc):
        self.graph.legend(title=title, loc=loc)

    @staticmethod
    def display_graph():
        plt.show()


# Plotting module
def plot():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    graph = Graph()
    # graph.set_index(1)
    #graph.set_labels(fontsize=[14, 10, 10])
    graph.plot(x, y)
    graph.display_graph()


if __name__ == "__main__":
    plot()


