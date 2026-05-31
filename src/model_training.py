import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import os

# Parameters
VOCAB_SIZE = 10000
MAX_LEN = 200

# Load Dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)

# Padding
X_train = pad_sequences(
    X_train,
    maxlen=MAX_LEN
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LEN
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Build Model
model = Sequential([
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=32,
        input_length=MAX_LEN
    ),

    SimpleRNN(32),

    Dense(
        1,
        activation="sigmoid"
    )
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Evaluation
loss, acc = model.evaluate(
    X_test,
    y_test
)

print("Test Accuracy:", acc)

# Save
os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    "models/rnn_model.keras"
)

print("Model Saved Successfully")