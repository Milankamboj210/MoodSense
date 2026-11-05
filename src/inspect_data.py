from tensorflow.keras.datasets import imdb
(X_train , y_train),(X_test , y_test) = imdb.load_data(num_words=10000)
# Examine review lengths
review_lengths = [len(review) for review in X_train]
# Get the dictionary that maps words to integer indices
word_index = imdb.get_word_index()

# Reverse it so we can map indices back to words
reverse_word_index = {
    value: key for key, value in word_index.items()
}

# Select the first review
sample_review = X_train[0]

# Convert token IDs back into words
decoded_review = " ".join([
    reverse_word_index.get(i - 3, "?")
    for i in sample_review
])

print("\nDecoded review:")
print(decoded_review)
print("\nSentiment label:", y_train[0])
