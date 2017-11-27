from keras.layers import Embedding, Dense, LSTM, Activation
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from preprocessing import *


def baseline_model(vocab_size, embed_length, lstm_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_length, mask_zero=True))
    model.add(LSTM(lstm_size, return_sequences=False))
    model.add(Dense(3, activation="sigmoid"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model


def split_data(df, ratio, state=1):
    """
    :param ratio: test fraction
    :param state: random_state
    :return:
    """
    X = df.text.values
    y = df.author.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=state)
    return X_train, X_test, y_train, y_test


def encode_authors(labels):
    """
    :param labels: np.array or list of string authors
    :return: categorical list (for keras)
    """
    authors_vocab = {"EAP": 0, "HPL": 1, "MWS": 2}
    y = [authors_vocab[label] for label in labels]
    y = to_categorical(y, num_classes=3)
    return y


def encode_texts(text, embedding_vocab, embedding_size):
    emb_func = lambda sent: sentence_to_emb(sent, embedding_vocab, embedding_size)
    emb_texts = np.array([emb_func(sent) for sent in text])
    return emb_texts

if __name__ == "__main__":

    train_df = create_df("train.csv")
    train_df.text = train_df.text.apply(clean_text)
    train_df.text = train_df.text.apply(lambda row: lemmatize_text(row))

    vocab_size = 5000
    vocab = get_vocabulary(train_df, length=vocab_size)
    emb_vocab = embedding_mapping(vocab)
    emb_vocab_size = len(emb_vocab)


    X_train, X_test, y_train, y_test = split_data(train_df, 0.8)
    embed_size = 64
    X_train = encode_texts(X_train, emb_vocab, embed_size)
    X_test = encode_texts(X_test, emb_vocab, embed_size)

    y_train = encode_authors(y_train)
    y_test = encode_authors(y_test)

    num_epochs = 3
    lstm_size = 50
    batch_size = 64


    model = baseline_model(emb_vocab_size, embed_size, lstm_size)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs)

    scores = model.evaluate(X_test, y_test)
    print("Accuracy:", scores[1])
