from database import Database
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from preprocessing import tokenize
from sklearn.cluster import KMeans
from readtxt import get_vocab

db = Database()


def main():
    df = db.find()

    vocab = get_vocab()

    count_vec = CountVectorizer(min_df=5, max_df=0.6,
                                # vocabulary=vocab,
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
    print(type(centroids), centroids.shape)
    terms = count_vec.get_feature_names()
    for i in range(5):
        print("\nCluster %d:" % i)
        print(", ".join(terms[index] for index in centroids[i, :10]))

    topic_values = km.predict(doc_term_matrix)
    print(topic_values.shape)

    df['Topic'] = topic_values
    df1 = df[['title', 'Topic']]
    print(df1.head(10))


if __name__ == "__main__":
    main()