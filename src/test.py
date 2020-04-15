import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from database import Database
from sklearn.decomposition import TruncatedSVD, NMF, PCA, LatentDirichletAllocation
import numpy as np
from preprocessing import tokenize
from plotting import Graph, plot_10_most_common_words
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.utils.extmath import randomized_svd
from scipy import linalg
from extraction import Extraction
from matplotlib import pyplot as plt
from readtxt import get_vocab

db = Database()
db.open(collection="mix_adverts")


def elbow_method(vector):
    # K means algorithm to find out the SSE
    print("############# Optimal K Value Algorithm Calculating...... ##############")
    sse = []
    loop = list(range(2, 30, 1))
    for i in loop:
        print("K =", i)
        model = KMeans(n_clusters=i, init='k-means++', random_state=122, max_iter=600)
        model.fit(vector)
        print("inertia for {} =".format(i), model.inertia_)
        sse.append(model.inertia_)

    graph = Graph()
    graph.set_labels("SSE vs K clusters", "n_cluster", "SSE")
    graph.plot(loop, sse)
    graph.display_graph()


def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data).groupby(clusters).mean()
    print(df.head(10))
    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))


def test_n_components(vector, terms):
    U, s, Vh = linalg.svd(vector, full_matrices=False)
    print(U.shape, s.shape, Vh.shape, terms.shape)

    print(sum(s > 2))
    graph = Graph()
    graph.plot(list(range(1, len(s) + 1)), s)
    graph.display_graph()

    num_top_words = 8

    def show_topics(a):
        top_words = lambda t: [terms[i] for i in np.argsort(t)[:-num_top_words - 1:-1]]
        topic_words = ([top_words(t) for t in a])
        return [' '.join(t) for t in topic_words]

    print(show_topics(Vh[:10]))


def extract_test_data(ext, key):
    ext.set_key(key)
    test_jobs = ext.get_job_titles(max_page=15)
    print("Extraction Finished")
    descriptions = [job.description for job in test_jobs]
    return descriptions


def perform_predict(vectorizer, model, data):
    predict_vector = vectorizer.transform(data)
    # test_u, test_s, test_vt = randomized_svd(predict_vector, n_components=1,
    #                                          n_iter=100,
    #                                          random_state=122)
    # svd = TruncatedSVD(n_components=1)
    # svd.fit(predict_vector)

    predictions = model.predict(predict_vector)
    count = [0] * 7
    for value in predictions:
        count[value] += 1
    top_2 = np.argsort(count)[-2:][::-1]
    return top_2


def main():
    df = db.find()

    vocab = get_vocab()

    count_vec = CountVectorizer(min_df=5, max_df=0.6,
                                vocabulary=vocab,
                                tokenizer=tokenize, lowercase=True)

    doc_term_matrix = count_vec.fit_transform(df.description)

    LDA = LatentDirichletAllocation(n_components=5, random_state=42)
    LDA.fit(doc_term_matrix)

    print("LDA")
    for i, topic in enumerate(LDA.components_):
        print(f'Top 10 words for topic #{i}:')
        print([count_vec.get_feature_names()[i] for i in topic.argsort()[-20:]])
        print('\n')

    topic_values = LDA.transform(doc_term_matrix)
    print(topic_values.shape)
    df['Topic'] = topic_values.argmax(axis=1)
    df1 = df[['title', 'Topic']]
    print(df1.head(10))

    print("LSA")
    transformer = TfidfTransformer()
    doc_term_matrix = transformer.fit_transform(doc_term_matrix)

    plot_10_most_common_words(doc_term_matrix, count_vec)

    LSA = TruncatedSVD(5, random_state=42)
    # normalizer = Normalizer(copy=False)
    # lsa = make_pipeline(LSA, normalizer)
    # X = lsa.fit_transform(doc_term_matrix)
    LSA.fit(doc_term_matrix)
    for i, topic in enumerate(LSA.components_):
        print(f'Top 10 words for topic #{i}:')
        print([count_vec.get_feature_names()[i] for i in topic.argsort()[-10:]])
        print('\n')

    topic_values = LSA.transform(doc_term_matrix)
    print(topic_values.shape)
    df['Topic'] = topic_values.argmax(axis=1)
    df1 = df[['title', 'Topic']]
    print(df1.head(10))

    print("Kmeans")
    km = KMeans(n_clusters=5, random_state=42, init='k-means++', max_iter=600)
    km.fit(doc_term_matrix)
    print("Top terms per cluster:")
    centroids = km.cluster_centers_.argsort()[:, ::-1]
    print(km.cluster_centers_)
    terms = count_vec.get_feature_names()
    for i in range(5):
        print("\nCluster %d:" % i)
        print(", ".join(terms[index] for index in centroids[i, :10]))

    topic_values = km.predict(doc_term_matrix)
    print(topic_values.shape)

    df['Topic'] = topic_values
    cluster = []
    for i in range(5):
        cluster.append(df[df.Topic == i])

    df1 = df[['title', 'Topic']]
    print(df1.head(10))
    print(cluster[0].shape)
    print(cluster[1].shape)
    print(cluster[2].shape)
    print(cluster[3].shape)
    print(cluster[4].shape)

    for i in range(5):
        matrix = count_vec.fit_transform(cluster[i].description)
        LDA.fit(matrix)
        print(f'LDA in cluster {i}')
        for idx, topic in enumerate(LDA.components_):
            print(f'Top 10 words for topic #{idx}:')
            print([count_vec.get_feature_names()[idx] for idx in topic.argsort()[-20:]])
            print('\n')

        print(f'SVD in cluster {i}')
        LSA.fit(matrix)
        for idx, topic in enumerate(LSA.components_):
            print(f'Top 10 words for topic #{idx}:')
            print([count_vec.get_feature_names()[idx] for idx in topic.argsort()[-10:]])
            print('\n')






    #
    # print("############# Vectorizing job description...... ##############")
    # vectorizer = TfidfVectorizer(tokenizer=tokenize,
    #                              vocabulary=vocab,
    #
    #                              use_idf=True, lowercase=True)
    # word_count_vector = vectorizer.fit_transform(df.description).todense().T
    #
    # print(word_count_vector)
    #
    # vectors_df = pd.DataFrame(word_count_vector, columns=vectorizer.get_feature_names())
    # print(vectors_df)
    # print(vectors_df.head(10))
    #
    # feature_array = np.array(vectorizer.get_feature_names())
    # print(feature_array.shape)
    # print(feature_array)
    # print("c#" in feature_array)
    # print("c++" in feature_array)
    # print("java" in feature_array)
    # print("php" in feature_array)
    # print("asp" in feature_array)
    # print(".net" in feature_array)
    # # n = 10
    # # top_n = feature_array[tfidf_sorting][:n]
    # # print(top_n)
    #
    # print(word_count_vector.shape)
    #
    # # Elbow Method
    # # elbow_method(word_count_vector)
    # optimal_k = 7
    #
    # # Latent Semantic Analysis (LSA)
    # # svd = TruncatedSVD(n_components=optimal_k)
    # # doc_topics = svd.fit_transform(word_count_vector)
    # # print(doc_topics.shape)
    # # print(svd.components_.shape)
    #
    # # test_n_components(word_count_vector, feature_array)
    #
    # # u, s, vt = randomized_svd(word_count_vector, n_components=optimal_k,
    # #                           n_iter=100,
    # #                           random_state=122)
    #
    # # nmf = NMF(n_components=optimal_k, init='nndsvda',
    # #           random_state=42,
    # #           ).fit(word_count_vector)
    # # print(nmf.components_.shape)
    #
    # # PCA TODO: Try to use PCA and see the result and predictions
    # # TODO: Or try to drag more data sample to see the different between clusters
    # # pca = PCA(n_components=100, random_state=42)
    # # out = pca.fit_transform(word_count_vector.todense())
    # # print(out.shape, pca.components_.shape)
    #
    # print("Performing dimensionality reduction using LSA")
    # # Vectorizer results are normalized, which makes KMeans behave as
    # # spherical k-means for better results. Since LSA/SVD results are
    # # not normalized, we have to redo the normalization.
    # svd = TruncatedSVD(100)
    # normalizer = Normalizer(copy=False)
    # lsa = make_pipeline(svd, normalizer)
    # X = lsa.fit_transform(word_count_vector)
    # print(X.shape)
    # explained_variance = svd.explained_variance_ratio_.sum()
    # print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
    #
    # print("################ Modeling Clusters ################")
    # km = KMeans(n_clusters=optimal_k, init='k-means++', random_state=122, max_iter=600)
    # km.fit(X)
    #
    # # # Get top N keywords
    # # # get_top_keywords(vt, clusters, vectorizer.get_feature_names(), 10)
    #
    # # Method 2
    # print("Top terms per cluster:")
    # centroids = km.cluster_centers_.argsort()[:, ::-1]
    # print(km.cluster_centers_.shape)
    # terms = vectorizer.get_feature_names()
    # print(len(terms))
    # for i in range(optimal_k):
    #     print("\nCluster %d:" % i)
    #     print(", ".join(terms[index] for index in centroids[i, :10]))
    #
    # # lists = ["web developer", "mobile app developer", "software engineer", "computer science", "IT", "network",
    # #          "software tester"]
    # # ext = Extraction('https://www.jobstreet.com.my/en/job-search/job-vacancy.php',
    # #                  {
    # #                     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36'
    # #                  }
    # #                  )
    # # for key in lists:
    # #     descriptions = extract_test_data(ext, key)
    # #     cluster = perform_predict(vectorizer, km, descriptions)
    # #     print("Cluster for %s" % key)
    # #     print(cluster)


if __name__ == "__main__":
    main()
