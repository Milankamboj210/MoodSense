
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

max_features = 10000
max_len = 500

word_index = imdb.get_word_index()
model = load_model("models/moodsense_model_v2.keras")

def encode_review(review):
    words = review.lower().split()
    encoded_review = []

    for word in words:
        word_id = word_index.get(word)

        if word_id is not None:
            encoded_id = word_id + 3

            if encoded_id >= max_features:
                encoded_id = 2
        else:
            encoded_id = 2

        encoded_review.append(encoded_id)

    return encoded_review

reviews = [
    "great",
    "This movie is great",
    "This movie was absolutely great and wonderful"
]

for review in reviews:
    encoded = encode_review(review)
    padded = pad_sequences([encoded], maxlen=max_len)

    score = model.predict(padded, verbose=0)[0][0]

    print(
        review,
        "->",
        round(float(score), 6),
        "->",
        "Positive" if score >= 0.5 else "Negative"
    )