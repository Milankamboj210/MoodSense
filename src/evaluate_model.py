
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

max_features = 10000
max_len = 500

(_, _), (X_test, y_test) = imdb.load_data(num_words=max_features)

X_test = pad_sequences(X_test, maxlen=max_len)

model = load_model("models/moodsense_model_v2.keras")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    batch_size=32
)

print("\nMoodSense V2 Test Results")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Correct predictions:", round(test_accuracy * len(y_test)))
print("Incorrect predictions:", len(y_test) - round(test_accuracy * len(y_test)))
