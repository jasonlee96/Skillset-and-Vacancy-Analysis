from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from plotting import Graph
import matplotlib.pyplot as plt
from database import Database
from readtxt import get_vocab
from preprocessing import tokenize
import numpy as np
import pandas as pd

N_CLUSTER = 11
vocab = get_vocab()
# MIX_ADVERT -> 0.1, SPECIFIED ADVERT 0.5
vectorizer = TfidfVectorizer(min_df=10, max_df=0.5,
                             use_idf=True,
                             #vocabulary=vocab,
                             smooth_idf=True,
                             sublinear_tf=True,
                             tokenizer=tokenize, lowercase=True)

def elbow_method(vector):
    # K means algorithm to find out the SSE
    print("############# Optimal K Value Algorithm Calculating...... ##############")
    sse = []
    scores = []
    sil = []
    loop = list(range(2, 30, 1))
    for i in loop:
        print("K =", i)
        model = KMeans(n_clusters=i, init='k-means++', random_state=122, max_iter=600)
        model.fit(vector)
        labels = model.labels_
        scores.append(model.score(vector))
        sil.append(silhouette_score(vector, labels, metric='euclidean'))
        print("inertia for {} =".format(i), model.inertia_)
        sse.append(model.inertia_)

    graph = Graph()
    graph.set_labels("SSE vs K clusters", "n_cluster", "SSE")
    graph.plot(loop, sse)
    graph.display_graph()
    graph = Graph()
    graph.set_labels("SSE vs K clusters", "n_cluster", "scores")
    graph.plot(loop, scores)
    graph.display_graph()
    graph = Graph()
    graph.set_labels("SSE vs K clusters", "n_cluster", "Silhouette")
    graph.plot(loop, sil)
    graph.display_graph()


def get_top_features_cluster(tf_idf_array, prediction, n_feats):
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction == label) # indices for each cluster
        x_means = np.mean(tf_idf_array[id_temp], axis=0) # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
        features = vectorizer.get_feature_names()
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns=['features', 'score'])
        dfs.append(df)
    cnt = 0
    for df in dfs:
        df.plot(kind='barh', x='features', y='score', title=("cluster " + str(cnt)))
        cnt += 1
        plt.show()
    return dfs


def main():
    db = Database()
    db.open(collection="mix_adverts")

    df = db.find_query(key="finance")
    print(df.head())
    # TODO: tune title or key and check w, w/o vocabs
    df['new_description'] = df.title + '\n' + df.description
    doc_term_matrix = vectorizer.fit_transform(df.new_description)
    tf_idf_norm = normalize(doc_term_matrix)
    tf_idf_array = tf_idf_norm.toarray()
    # print(vectorizer.stop_words_)

    pca = PCA(n_components=2)
    pca.fit(tf_idf_array)
    data2D = pca.transform(tf_idf_array)
    plt.scatter(data2D[:,0], data2D[:,1])
    plt.show()              #not required if using ipython notebook

    red_dim = TruncatedSVD(n_components=100, random_state=122)
    red_data = red_dim.fit_transform(tf_idf_array)

    # elbow_method(tf_idf_array)

    kmeans = KMeans(n_clusters=N_CLUSTER, random_state=122, init='k-means++', max_iter=600)
    value = kmeans.fit_predict(red_data)
    centers2D = pca.fit_transform(red_data)
    centroid = pca.transform(kmeans.cluster_centers_)
    #plt.hold(True)
    plt.scatter(centers2D[:, 0], centers2D[:, 1],
                c=value)
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', s=200, c='r', linewidths=3)
    plt.show()  # not required if using ipython notebook

    print(red_data.shape)
    print(red_dim.components_.shape)
    print(kmeans.cluster_centers_.shape)
    combine_weight = np.dot(kmeans.cluster_centers_, red_dim.components_)
    print(combine_weight.shape)

    print("Top terms per cluster:")
    combine_weight = np.abs(combine_weight)
    terms = vectorizer.get_feature_names()
    cnt = 0
    for i in range(kmeans.n_clusters):
        top5 = np.argsort(combine_weight[i])[-20:]
        # keywords = zip([terms[j] for j in top5], combine_weight[i, top5])
        best_features = zip([terms[j] for j in top5], combine_weight[i, top5])
        df = pd.DataFrame(best_features, columns=['features', 'score'])
        print(df.head())
        df.plot(kind='barh', x='features', y='score', title=("cluster " + str(cnt)))
        cnt += 1
        plt.show()

    # dfs = get_top_features_cluster(red_data, value, 15)
    test_term = ["web developer", "software engineer", "mobile app developer", "marketing", "accountant",
                 "software tester", "game developer", "designer", "network engineer", "tech lead", "finance",
                 "data scientist"]
    test_vec = vectorizer.transform(test_term)
    test_vec = red_dim.transform(test_vec)
    test_val = kmeans.predict(test_vec)
    print(test_val)
    result = zip(test_term, test_val)
    print(list(result))


if __name__ == "__main__":
    main()