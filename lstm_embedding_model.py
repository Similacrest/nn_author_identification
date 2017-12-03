from keras.layers import Embedding, Dense, LSTM, Activation, Dropout, Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from preprocessing import *


def baseline_model(vocab_size, embed_length, lstm_size, filters, kernel_size, pool_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_length, mask_zero=False))
    model.add(Dropout(0.4))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_size, recurrent_dropout=0.4))
    model.add(Dense(3, activation="sigmoid"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    print(model.summary())
    return model


if __name__ == "__main__":

    train_df = create_df("train.csv")
    train_df.text = train_df.text.apply(clean_text)
    train_df.text = train_df.text.apply(lambda row: lemmatize_text(row))

    vocab_size = 5000
    embed_size = 64
    num_epochs = 5
    lstm_size = 50
    batch_size = 64
    kernel_size = 4
    filters = 64
    pool_size = 4

    vocab = get_vocabulary(train_df, length=vocab_size)
    emb_vocab = embedding_mapping(vocab)
    emb_vocab_size = len(emb_vocab)

    X_train, X_test, y_train, y_test = split_data(train_df, 0.8)

    X_train = encode_texts(X_train, emb_vocab, embed_size)
    X_test = encode_texts(X_test, emb_vocab, embed_size)

    y_train = encode_authors(y_train)
    y_test = encode_authors(y_test)


    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    model = baseline_model(emb_vocab_size, embed_size, lstm_size, filters, kernel_size, pool_size)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs)

    scores = model.evaluate(X_test, y_test)
    print("Accuracy:", scores[1])
