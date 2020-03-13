import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from database import Database
from sklearn.decomposition import TruncatedSVD
import numpy as np
from preprocessing import tokenize


db = Database()
db.open()

n_comp = 10


def display_matrix(encoding_matrix, topic):
    print(topic)
    print(encoding_matrix.sort_values('abs_'+topic, ascending=False).head(10))


def main():
    df = db.find()

    vectorizer = TfidfVectorizer(tokenizer=tokenize, use_idf=True, max_df=0.40, min_df=10)
    vectors = vectorizer.fit_transform(df.description)
    vectors_df = pd.DataFrame(vectors.todense(), columns=vectorizer.get_feature_names())
    print(vectors_df)
    #print(vectors_df.head(10))
    svd = TruncatedSVD(n_components=n_comp)
    lsa = svd.fit_transform(vectors)

    # vectors_df = pd.DataFrame(lsa, columns=["topic_1", "topic_2"])
    # vectors_df['body'] = df.prepared_description
    # print(vectors_df[['body','topic_1', 'topic_2']])

    feature_names = vectorizer.get_feature_names()
    print(feature_names)
    index_list = ['topic_'+str(i) for i in range(1, n_comp+1)]
    encoding_matrix = pd.DataFrame(svd.components_, index=index_list,
                                   columns=feature_names).T

    for cnt in range(1, n_comp + 1):
        encoding_matrix['abs_topic_'+str(cnt)] = np.abs(encoding_matrix['topic_'+str(cnt)])
    for cnt in range(1, n_comp + 1):
        display_matrix(encoding_matrix, 'topic_'+str(cnt))


if __name__ == "__main__":
    main()