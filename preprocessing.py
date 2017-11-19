import nltk
import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer


# Read csv file and return Pandas DataFrame
def create_df(filename):
    df = pd.read_csv(filename)
    return df


# Delete punctuation and stopwords
def clean_text(text):
    stopwords = nltk.corpus.stopwords.words("english")
    lowercase_text = text.lower()
    tokenize_text = nltk.word_tokenize(lowercase_text)
    text_without_stopwords = [w for w in tokenize_text if w not in stopwords]
    pure_txt = [w for w in text_without_stopwords if w not in string.punctuation]
    return " ".join(pure_txt)


# Convert word to its normal form
def lemmatize_text(text, stem=False):
    """
    :param stem: If True - use stemming, False - use lemmatization. Defalut: lemmatizetion
    """
    text = text.split()
    if stem:
        stemmer = nltk.stem.PorterStemmer()
        normal_txt = [stemmer.stem(w) for w in text]
    else:
        lemmanizer = nltk.stem.WordNetLemmatizer()
        normal_txt = [lemmanizer.lemmatize(w) for w in text]
    return " ".join(normal_txt)

# From all the text create vocabuluary
def get_vocabulary(df, length=3000):
    """
    length - the number of the most frequent words to create a vocabulary, other - ignore.
    """
    docs = df.text.values
    vec = CountVectorizer(max_features=length)
    vec.fit_transform(docs)
    return vec.vocabulary_


# Text to matrix, return list
def vectorize_sentence(sentenses,  vocabulary):
    """
    Using Bi-gram models
    sentenses - list of lists
    """
    vectorizer = CountVectorizer(vocabulary=vocabulary, ngram_range=(2,2))
    sentence_transform = vectorizer.fit_transform(sentences)
    return sentence_transform.toarray()


if __name__ == "__main__":
    train_df = create_df("train.csv")

    train_df.text = train_df.text.apply(clean_text)
    train_df.txt = train_df.text.apply(lambda row: lemmatize_text(row))

    vocab = get_vocabulary(train_df)

    example = train_df.text.values[0:2]

    vectorized = vectorize_sentence(example, vocab)
    print(vectorized.shape)





