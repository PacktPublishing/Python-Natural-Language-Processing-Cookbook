import pandas as pd
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from keras import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import load_model
import matplotlib.pyplot as plt
from Chapter04.preprocess_bbc_dataset import get_data
from Chapter04.keyword_classification import get_labels
from Chapter04.preprocess_bbc_dataset import get_stopwords
from Chapter04.svm_classification import create_dataset, new_example

MAX_NUM_WORDS = 50000
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 300

bbc_dataset = "Chapter04/bbc-text.csv"

def evaluate(model, X_test, Y_test, le):
    Y_pred = model.predict(X_test)
    Y_pred = Y_pred.argmax(axis=-1)
    Y_test = Y_test.argmax(axis=-1)
    Y_new_pred = [le.inverse_transform([value]) for value in Y_pred]
    Y_new_test = [le.inverse_transform([value]) for value in Y_test]
    print(classification_report(Y_new_test, Y_new_pred))

def plot_model(history):
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

def create_tokenizer(input_data, save_path):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(input_data)
    save_tokenizer(tokenizer, save_path)
    return tokenizer

def save_tokenizer(tokenizer, filename):
    with open(filename, 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer(filename):
    with open(filename, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer 

def transform_text(tokenizer, input_text):
    if (isinstance(tokenizer, str)):
        tokenizer = load_tokenizer(tokenizer)
    X_input = tokenizer.texts_to_sequences(input_text)
    X_input = pad_sequences(X_input, maxlen=MAX_SEQUENCE_LENGTH)
    return X_input

def train_model(df, le):
    tokenizer = create_tokenizer(df['text'].values, 'Chapter04/bbc_tokenizer.pickle')
    X = transform_text(tokenizer, df['text'].values)
    Y = pd.get_dummies(df['label']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
    model = Sequential()
    optimizer = tf.keras.optimizers.Adam(0.001)
    model.add(Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation='softmax'))
    loss='categorical_crossentropy' #Standard for multiclass classification
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    epochs = 7
    batch_size = 64
    es = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[es])
    accr = model.evaluate(X_test,Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    model.save('Chapter04/bbc_model_scratch1.h5')
    evaluate(model, X_test, Y_test, le)
    plot_model(history)

def load_and_evaluate_existing_model(model_path, tokenizer_path, df, le):
    model = load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)
    X = transform_text(tokenizer, df['text'].values)
    Y = pd.get_dummies(df['label']).values
    evaluate(model, X, Y, le)

def test_new_example(model, tokenizer, le, text_input):
    X_example = transform_text(tokenizer, [new_example])
    label_array = model.predict(X_example)
    new_label = np.argmax(label_array, axis=-1)
    print(new_label)
    print(le.inverse_transform(new_label))

def main():
    data_dict = get_data(bbc_dataset)
    le = get_labels(list(data_dict.keys()))
    df = create_dataset(data_dict, le)
    train_model(df, le)
    #load_and_evaluate_existing_model('Chapter04/bbc_model.h5', 'Chapter04/bbc_tokenizer.pickle', df, le)
    #model = load_model('Chapter04/bbc_model.h5')
    #tokenizer = 'Chapter04/bbc_tokenizer.pickle'
    #test_new_example(model, tokenizer, le, new_example)

if (__name__ == "__main__"):
    main()

