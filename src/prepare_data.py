from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=10000
)
X_train_padded = pad_sequences(
    X_train,
    maxlen=500
)
X_test_padded = pad_sequences(
    X_test,
    maxlen=500
)
print("Original first review length:", len(X_train[0]))
print("Padded first review shape:", X_train_padded[0].shape)
print("Training data shape:", X_train_padded.shape)
print("Testing data shape:", X_test_padded.shape)