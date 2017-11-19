import nltk
import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import *
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Embedding, LSTM
from keras.models import Model

def getData(train_df):
    train_df.text = train_df.text.apply(clean_text)
    train_df.text = train_df.text.apply(lambda row: lemmatize_text(row))

    vocab = get_vocabulary(train_df, 3000)

    X = vectorize_sentence(train_df.text.values, vocab)

    authors_vocab = {"EAP" : 0, "HPL" : 1, "MWS" : 2}
    y = []
    for i in range(len(X)):
        y.append(authors_vocab[train_df['author'][i]])
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=33)

    return (X_train, X_test, y_train, y_test, len(vocab))

def trainModel(train_df):
    X_train, X_test, y_train, y_test, vocab_size = getData(train_df)

    embed_size = 128
    x = Input(shape=(None,), dtype='int32')
    e = Embedding(vocab_size, embed_size, mask_zero=True)(x)
    r = LSTM(30, return_sequences=False)(e)
    p = Dense(1, activation='sigmoid')(r)

    lstm_model = Model(x, p)
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = lstm_model.fit(X_train, y_train, batch_size=32, epochs=1, validation_data=(X_test, y_test))

    return history

if __name__ == "__main__":

    train_df = create_df("train.csv")

    history = trainModel(train_df)
    print(history)
