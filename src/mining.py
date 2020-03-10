import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from database import Database
from sklearn.decomposition import TruncatedSVD
import numpy as np

db = Database()
db.open()


def main():
    df = db.find({"key": "software engineer"})

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(df.prepared_description)

    svd = TruncatedSVD(n_components=5)
    lsa = svd.fit_transform(vectors)

    # vectors_df = pd.DataFrame(lsa, columns=["topic_1", "topic_2"])
    # vectors_df['body'] = df.prepared_description
    # print(vectors_df[['body','topic_1', 'topic_2']])

    feature_names = vectorizer.get_feature_names()
    print(feature_names)
    encoding_matrix = pd.DataFrame(svd.components_, index=['topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5'],
                                   columns=feature_names).T
    print(encoding_matrix)

    encoding_matrix['abs_topic_1'] = np.abs(encoding_matrix['topic_1'])
    encoding_matrix['abs_topic_2'] = np.abs(encoding_matrix['topic_2'])
    encoding_matrix['abs_topic_3'] = np.abs(encoding_matrix['topic_3'])
    encoding_matrix['abs_topic_4'] = np.abs(encoding_matrix['topic_4'])
    encoding_matrix['abs_topic_5'] = np.abs(encoding_matrix['topic_5'])
    print("Topic 1")
    print(encoding_matrix.sort_values('abs_topic_1', ascending=False).head(10))
    print("Topic 2")
    print(encoding_matrix.sort_values('abs_topic_2', ascending=False).head(10))
    print("Topic 3")
    print(encoding_matrix.sort_values('abs_topic_3', ascending=False).head(10))
    print("Topic 4")
    print(encoding_matrix.sort_values('abs_topic_4', ascending=False).head(10))
    print("Topic 5")
    print(encoding_matrix.sort_values('abs_topic_5', ascending=False).head(10))


if __name__ == "__main__":
    main()