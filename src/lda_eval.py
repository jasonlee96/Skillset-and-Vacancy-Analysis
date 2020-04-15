from database import Database
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans
from preprocessing import tokenize
from readtxt import get_vocab
from plotting import Graph, plot_10_most_common_words
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


db = Database()
db.open(collection="mix_adverts")


def main():
    df = db.find()

    vocab = get_vocab()

    count_vec = CountVectorizer(min_df=5, max_df=0.6,
                                vocabulary=vocab,
                                tokenizer=tokenize, lowercase=True)

    doc_term_matrix = count_vec.fit_transform(df.description)

    perplexities5 = []
    perplexities7 = []
    perplexities9 = []
    scores5 = []
    scores7 = []
    scores9 = []
    loop = list(range(2, 20, 1))
    search_params = {'n_components': loop, 'learning_decay': [.5, .7, .9]}
    lda = LatentDirichletAllocation(random_state=122, learning_method='online', n_components=4, learning_decay=0.9)
    # model = GridSearchCV(cv=None, error_score='raise', estimator=lda, param_grid=search_params)
    # model.fit(doc_term_matrix)
    #
    # # Best Model
    # best_lda_model = model.best_estimator_
    #
    # # Model Parameters
    # print("Best Model's Params: ", model.best_params_)
    #
    # # Log Likelihood Score
    # print("Best Log Likelihood Score: ", model.best_score_)
    #
    # # Perplexity
    # print("Model Perplexity: ", best_lda_model.perplexity(doc_term_matrix))

    lda_output = lda.fit_transform(doc_term_matrix)

    testdf = pd.DataFrame(lda.components_, columns=count_vec.get_feature_names())
    print(testdf.head(20))
    # TODO: use lda output to perform kmeans
    # TODO: and submatrix the document and perform kmeans to inner cluster

    print("LDA")
    for i, topic in enumerate(lda.components_):
        print(topic)
        print(f'Top 10 words for topic #{i}:')
        print([count_vec.get_feature_names()[i] for i in topic.argsort()[-20:]])
        print('\n')
    # search_params = {'n_components': loop, 'batch_size': [128, 256, 512]}
    # lda = LatentDirichletAllocation(random_state=122, learning_method='batch', n_components=3)
    # model = GridSearchCV(cv=None, error_score='raise', estimator=lda, param_grid=search_params)
    # model.fit(doc_term_matrix)
    #
    # # Best Model
    # best_lda_model = model.best_estimator_
    #
    # # Model Parameters
    # print("Best Model's Params: ", model.best_params_)
    #
    # # Log Likelihood Score
    # print("Best Log Likelihood Score: ", model.best_score_)
    #
    # # Perplexity
    # print("Model Perplexity: ", best_lda_model.perplexity(doc_term_matrix))
    # lda.fit(doc_term_matrix)
    # print("LDA")
    # for i, topic in enumerate(lda.components_):
    #     print(f'Top 10 words for topic #{i}:')
    #     print([count_vec.get_feature_names()[i] for i in topic.argsort()[-20:]])
    #     print('\n')
    km = KMeans(n_clusters=4, random_state=122, init='k-means++', max_iter=600)
    clusters = km.fit_predict(lda_output)

    # print("Top terms per cluster:")
    centroids = km.cluster_centers_.argsort()[:, ::-1]
    print(km.cluster_centers_)
    print(centroids)
    # terms = count_vec.get_feature_names()
    # for i in range(5):
    #     print("\nCluster %d:" % i)
    #     print(", ".join(terms[index] for index in centroids[i, :10]))

    # Build the Singular Value Decomposition(SVD) model
    svd_model = TruncatedSVD(n_components=2)  # 2 components
    lda_output_svd = svd_model.fit_transform(lda_output)

    # X and Y axes of the plot using SVD decomposition
    x = lda_output_svd[:, 0]
    y = lda_output_svd[:, 1]

    # Weights for the 15 columns of lda_output, for each component
    print("Component's weights: \n", np.round(svd_model.components_, 2))

    # Percentage of total information in 'lda_output' explained by the two components
    print("Perc of Variance Explained: \n", np.round(svd_model.explained_variance_ratio_, 2))
    # Plot
    plt.figure(figsize=(12, 12))
    plt.scatter(x, y, c=clusters)
    plt.title("Segregation of Document-Topic Clusters", )
    plt.legend(['0', '1', '2', '3'])

    test_words = ["software engineer"]
    matrix = count_vec.transform(test_words)
    test_out = lda.transform(matrix)
    prediction = km.predict(test_out)
    print(prediction)

    lda_output_svd = svd_model.transform(test_out)

    # X and Y axes of the plot using SVD decomposition
    print(lda_output_svd[0, 0], lda_output_svd[0, 1])
    x = lda_output_svd[0][0]
    y = lda_output_svd[0][1]
    plt.scatter(x, y, c='r')
    plt.show()

    clusters = km.fit_predict(lda.components_.T)
    lda_output_svd = svd_model.fit_transform(lda.components_.T)
    x = lda_output_svd[:, 0]
    y = lda_output_svd[:, 1]
    print(x)
    print(y)
    print(clusters)
    plt.figure(figsize=(12, 12))
    plt.scatter(x, y, c=clusters)
    plt.title("Segregation of Term-Topic Clusters", )
    plt.show()
    # plot_10_most_common_words(doc_term_matrix, count_vec)


if __name__ == "__main__":
    main()
