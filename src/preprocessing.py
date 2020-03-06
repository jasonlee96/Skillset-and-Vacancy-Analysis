from database import Database
import nltk
from nltk.tokenize import MWETokenizer

db = Database()
# TODO: download nltk stopwords
stopwords = []
# Multi word expression that used to merge the separated token
mwe_list = [("c", "#"), (".", "net")]


def main():
    db.open()
    # DataFrame Sample
    #       _id     title               href        description     key
    # 0    ...      Software Engineer   URL         <descriptions>  software engineer
    df = db.find({"key": "software engineer"})
    # tokenizer = RegexpTokenizer(r'\w+')

    # Retokenize by merge those keywords from different tokens such as ('c', '#') -> ('c#')
    retokenizer = MWETokenizer(mwes=mwe_list, separator='')

    for index, row in df.iterrows():
        print(row['description'].lower())
        tokens = nltk.word_tokenize(row['description'].lower())
        print(retokenizer.tokenize(tokens))

    # TODO: process stopword with nltk list


if __name__ == "__main__":
    main()
