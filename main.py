from datasetReader import read_dataset
from embedding import embedding_layer
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPool1D
from matplotlib import pyplot as plt

top_words = 20000
max_tweet_length = 30
X_train, y_train, X_test, y_test, word_index = read_dataset('trainingandtestdata/training.1600000.processed.noemoticon.csv',
                                                'trainingandtestdata/testdata.manual.2009.06.14.csv', 1000000, max_tweet_length)
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)



# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

embedding_vector_length = 32

embedding = embedding_layer(word_index)

model = Sequential()
# model.add(Embedding(top_words, embedding_vector_length, input_length=max_tweet_length))
model.add(embedding)
model.add(BatchNormalization())
model.add(Conv1D(64, kernel_size=3, padding='same', activation='sigmoid'))
model.add(Dropout(0.4))
model.add(Conv1D(32, kernel_size=3, padding='same', activation='sigmoid'))
model.add(Dropout(0.2))
model.add(MaxPool1D())
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(80))
model.add(Dense(50, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=15, batch_size=128, verbose=2)

model.summary()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
