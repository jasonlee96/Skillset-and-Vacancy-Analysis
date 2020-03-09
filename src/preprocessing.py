from database import Database
import nltk
from nltk.tokenize import MWETokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Initialize Database object
db = Database()

# Lemmatizer initialization
lemmatizer = WordNetLemmatizer()

# list of symbols have to remove from the tokens list
symbols = ['*', '(', ')', ':', '-', ',', '.', '!', '?', '_', '/', ';', "'", "[", "]", '{', '}', '+', '&', '%', '#', '@',
           "<", '>', '|']
# stopwords list initialization
stopwords_list = stopwords.words("english")
stopwords_list.extend(symbols)

# Multi word expression that used to merge the separated token
mwe_list = [("c", "#"), (".", "net")]

# TODO: Remove line below after completed
# Word that being removed from tokens
removed_token = []


# function check_unicode
# params: word : str
# description: remove those unicode that can't encode into ascii code, such as chinese word and some special character
# return bool
def check_unicode(word) -> bool:
    try:
        word.encode('ascii').decode('utf-8')
        return True
    except UnicodeEncodeError:
        return False
    except UnicodeDecodeError:
        return False


# function get_pos
# params: word : str
# description: to get the pos tag of a word. Such as adj, verb, noun, adverb
# return wordnet object
def get_pos(word) -> wordnet:
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_maps = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }
    return tag_maps.get(tag, wordnet.NOUN)  # get pos tag with default value of Noun.


# function process_multiple
# params: word_list : list[str]
# description: remove stopword and lemmatize the word into its original form from the list
# return list of processed word
def process_multiple(word_list) -> list:
    # Removed token appending.
    # TODO: Remove line below after completed
    removed_token.extend([word for word in word_list if word not in stopwords_list])
    # Remove stopwords from the list
    filtered_list = [word for word in word_list if word not in stopwords_list]
    # Remove empty string in the token list
    filtered_list = filter(None, filtered_list)
    result = [lemmatizer.lemmatize(w, pos=get_pos(w)) for w in filtered_list]
    return result


# function process_single
# params: word : str
# description: remove stopword and lemmatize the word into its original form from a string
# return string or None
def process_single(word) -> str:
    if word not in stopwords_list:
        return lemmatizer.lemmatize(word, pos=get_pos(word))
    # TODO: Remove line below after completed
    removed_token.append(word)


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
        tokens = retokenizer.tokenize(nltk.word_tokenize(row['description'].lower()))
        prepared_token = []
        for word in tokens:
            if not check_unicode(word):
                # TODO: Remove line below after completed
                removed_token.append(word)
                # print("unicode detected", word)
                continue
            # Try to split the joined tokens such as ("Design/Test") into two different token
            if "/" in word and not word.startswith("/") and not word.endswith("/"):
                # print("circuited", word.split("/"), sep=" ")
                if "www" in word:
                    continue
                prepared_token.extend(process_multiple(word.split("/")))
                continue
            process_word = process_single(word)
            if process_word is not None:
                prepared_token.append(process_word)
        db.update_with_id(row['_id'], {"prepared_description": prepared_token})
    print("Number of removed token: ", len(removed_token))
    print(removed_token)


if __name__ == "__main__":
    main()
