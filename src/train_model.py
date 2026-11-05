
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

max_features = 10000
max_len = 500

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

model = Sequential([
    Input(shape=(max_len,)),
    Embedding(input_dim=max_features, output_dim=128, mask_zero=True),
    SimpleRNN(128, activation="tanh"),
    Dense(1, activation="sigmoid")
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

model.save("models/moodsense_model_v2.keras")

print("MoodSense V2 saved successfully!")
