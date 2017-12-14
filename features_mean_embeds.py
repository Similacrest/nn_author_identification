from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import nltk
import string
from keras.preprocessing.text import Tokenizer
from preprocessing import *
import numpy as np

def loadGlove(filename):
    emb_dict = {}
    with open(filename, 'r', encoding="utf8") as emb_file:
        for line in emb_file.readlines():
            row = line.strip().split(' ')
            emb_dict[row[0]] = row[1:]
    emb_file.close()
    return emb_dict

def make_embed_matrix(sents, emb_dict):
    """
    :param sents: training sentences as list
    :param emb_dict: dictionary (keys: words, values: corresponding embedding vector)
    :return:
    """
    tokenizer = Tokenizer(num_words=100000)
    tokenizer.fit_on_texts(sents)
    word_index = tokenizer.word_index
    emb_matrix = []
    for sent in sents:
        sent_vector = np.zeros((1, 50))
        i = 0
        for word in sent.split(' '):
            if word in word_index.keys() and emb_dict.get(word) is not None:
                curr_emb = np.array(emb_dict.get(word)).astype(np.float)
                sent_vector += curr_emb
                i += 1
            else:
                print("nixua")
        sent_vector /= i
        emb_matrix.append(sent_vector)
    # emb_matrix = np.stack(emb_matrix, axis=-1)
    return emb_matrix

def create_submission(preds, filename, ids):
    """
    :param preds: predictions
    :param filename: target csv file name
    :param ids: test ids
    :return:
    """
    preds_data = {"id": [],
                  "EAP": preds[:, 0],
                  "HPL": preds[:, 1],
                  "MWS": preds[:, 2]}
    df = pd.DataFrame(preds_data, columns=["EAP", "HPL", "MWS"], index=ids)
    df.to_csv(filename)

def get_additional_features(data):
    new_df = data.copy()
    eng_stopwords = set(nltk.corpus.stopwords.words("english"))
    new_df["words"] = new_df["text"].apply(lambda text: text.split())

    # Num words
    new_df["num_words"] = new_df["words"].apply(lambda words: len(words))

    # Num unique words
    new_df["num_unique_words"] = new_df["words"].apply(lambda words: len(set(words)))

    # Num stopwords
    new_df["num_stopwords"] = new_df["words"].apply(lambda words: len([w for w in words if w in eng_stopwords]))

    # Num punctuation
    new_df["num_punctuations"] = new_df["text"].apply(lambda text: len([c for c in text if c in string.punctuation]))

    # Num words upper
    new_df["num_words_upper"] = new_df["words"].apply(lambda words: len([w for w in words if w.isupper()]))

    # Num words title
    new_df["num_words_title"] = new_df["words"].apply(lambda words: len([w for w in words if w.istitle()]))

    # Mean word length
    new_df["mean_word_len"] = new_df["words"].apply(lambda words: np.mean([len(w) for w in words]))
    del new_df["words"]
    return new_df


if __name__ == "__main__":
    train_df = create_df("train")
    extended_train = get_additional_features(train_df)
    extended_train.text = extended_train.text.apply(clean_text)
    extended_train.text = extended_train.text.apply(lambda row: lemmatize_text(row))

    glove_embeds = loadGlove("glove.6B.50d.txt")
    emb_matrix_train = make_embed_matrix(extended_train.text.values, glove_embeds)

    authors_vocab = {"EAP": 0, "HPL": 1, "MWS": 2}
    y = []
    for i in range(len(train_df.author)):
        y.append(authors_vocab[train_df.author[i]])
    y = np.array(y)

    text_clf_svm = Pipeline([('clf-svm', SGDClassifier(loss='log', penalty='l2',
                                alpha=1e-4, n_iter=15, random_state=33)),
                             ])
    X = emb_matrix_train

    text_clf_svm.fit(X, y)

    test_df = create_df("test")
    extended_test = get_additional_features(test_df)
    extended_test.text = extended_test.text.apply(clean_text)
    extended_test.text = extended_test.text.apply(lambda row: lemmatize_text(row))

    emb_matrix_test = make_embed_matrix(extended_test.text.values, glove_embeds)
    X_pred = extended_test

    result = text_clf_svm.predict_proba(X_pred)
    ids_ = test_df.id

    create_submission(result, "nb_embeds_preds.csv", ids_)
