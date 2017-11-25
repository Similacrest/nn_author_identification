from keras.layers import Embedding, Dense, LSTM, Activation
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from preprocessing import *

def baseline_model(vocab_size, embed_length, lstm_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_length, mask_zero=True))
    model.add(LSTM(lstm_size, return_sequences=False))
    model.add(Dense(3, activation="sigmoid"))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
    print(model.summary())

    return model

def split_data(data, p):
    num_train = int(len(data)*p)
    train_set = data[:num_train]
    dev_set = data[num_train:]
    return train_set, dev_set

def author_to_int(labels):
    authors_vocab = {"EAP": 0, "HPL": 1, "MWS": 2}
    y = []
    for i in range(len(labels)):
        y.append(authors_vocab[labels[i]])
    return np.array(y)


if __name__ == "__main__":
    num_epochs = 3
    vocab_size = len(emb)
    embed_size = 64
    lstm_size = 50
    batch_size = 64

    model = baseline_model(vocab_size, embed_size, lstm_size)
    train, dev = split_data(train_df, 0.8)

    x_train = train.apply(lambda row: sentence_to_emb(row.to_string(), emb, embed_size))
    x_dev = dev.apply(lambda row: sentence_to_emb(row.to_string(), emb, embed_size))

    y_train = train.author.values
    y_dev = dev.author.values

    print(train.iloc[0], "train")
    print(x_train.iloc[0], "train")

    y_bin_train = author_to_int(y_train)
    y_bin_dev =  author_to_int(y_dev)

    y_bin_train = to_categorical(y_bin_train, num_classes=3)
    y_bin_dev = to_categorical(y_bin_dev, num_classes=3)

    print(y_bin_train)

    print(x_dev.shape, y_dev.shape)
    print(x_train.shape, y_train.shape)

    model.fit(x_train, y_bin_train, batch_size=batch_size,
              epochs=num_epochs, validation_data=(x_dev, y_bin_dev))
    scores = model.evaluate(x_dev, y_bin_dev, batch_size=batch_size)
