import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Activation

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

def r_squared(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Wiki Link: https://en.wikipedia.org/wiki/Coefficient_of_determination
    """
    res_sum_squares = tf.math.reduce_sum(tf.math.square(y_true - y_pred))
    y_mean = tf.math.reduce_mean(y_true)
    total_sum_squares = tf.reduce_sum(tf.math.square(y_true - y_mean))
    r2 = 1.0 - tf.math.divide(res_sum_squares, total_sum_squares)
    return tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)


def get_dataset() -> (
    tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
):
    dataset = load_diabetes()
    x: np.ndarray = dataset.data
    y: np.ndarray = dataset.target.reshape(-1, 1)
    x_train_, x_test_, y_train_, y_test_ = train_test_split(x, y, test_size=0.3)
    x_train: np.ndarray = x_train_.astype(np.float32)
    x_test: np.ndarray = x_test_.astype(np.float32)
    y_train: np.ndarray = y_train_.astype(np.float32)
    y_test: np.ndarray = y_test_.astype(np.float32)
    return (x_train, y_train), (x_test, y_test)


def build_model(num_features: int, num_targets: int) -> Sequential:
    model = Sequential()
    model.add(Dense(units=150, input_shape=(num_features,))) # Add the first layer with 10 units and the input size of num_features
    model.add(Activation("relu")) # Add the ReLU activation function for non-linearity
    model.add(Dense(units=100)) # Add the second layer with 8 units
    model.add(Activation("relu")) # Add the ReLU activation function for non-linearity
    model.add(Dense(units=50)) # Add the third layer with 6 units
    model.add(Activation("relu")) # Add the ReLU activation function for non-linearity
    model.add(Dense(units=num_targets)) # Add the output layer with num_targets units
    model.summary() # Print the model summary to the console

    return model


def main() -> None:
    (x_train, y_train), (x_test, y_test) = get_dataset()

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"num train samples: {x_train.shape[0]}")

    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"num test samples: {x_test.shape[0]}")

    num_features = x_train.shape[1]
    num_targets = y_train.shape[1]

    model = build_model(num_features=num_features, num_targets=num_targets) # Instantiate the model

    opt = Adam(learning_rate=0.01) # Instantiate the optimizer

    model.compile(loss="mse", optimizer=opt, metrics=[r_squared]) # Compile the model

    model.fit(
        x=x_train,
        y=y_train,
        epochs=5000,
        batch_size=128,
        verbose=1, # 0: silent, 1: progress bar, 2: one line per epoch
        validation_data=(x_test, y_test),
    ) # Train the model

    scores = model.evaluate(x_test, y_test) # Evaluate the model on the test data
    print(f"Test loss: {scores[0]}")
    print(f"Test R^2: {scores[1]}")

    prediction = model.predict(x_test[:5]) # Predict the first 5 samples in the test data
    print(f"Prediction:\n{prediction}")
    print(f"True values:\n{y_test[:5]}")

if __name__ == "__main__":
    main()
