from typing import Tuple

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

def get_dataset() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    dataset = load_diabetes()
    x = dataset.data
    y = dataset.target.reshape(-1, 1) # Reshape to 2D array
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # Convert to float32 for TensorFlow
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    return (x_train, y_train), (x_test, y_test) # return tuple of tuples

def main() -> None:
    (x_train, y_train), (x_test, y_test) = get_dataset()

    # Print shapes of the training and testing data
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")


if __name__ == "__main__":
    main()
