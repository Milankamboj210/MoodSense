import tensorflow as tf
embedding = tf.keras.layers.Embedding(input_dim = 10 , output_dim = 4)
sample = tf.constant([
    [1, 2, 3],
    [4, 5, 0]
])
result = embedding(sample)

print("Input shape:", sample.shape)
print("Output shape:", result.shape)
print("\nEmbedding output:")
print(result.numpy())
