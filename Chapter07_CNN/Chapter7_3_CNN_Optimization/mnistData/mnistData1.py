import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical


class MNIST:
    def __init__(
        self,
    ) -> None:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # Preprocess x data
        self.x_train = x_train.astype(np.float32)
        self.x_train = np.expand_dims(x_train, axis=-1)
        self.x_test = x_test.astype(np.float32)
        self.x_test = np.expand_dims(x_test, axis=-1)
        # Dataset attributes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.img_shape = (self.width, self.height, self.depth)
        self.num_classes = 10
        # Preprocess y data
        self.y_train = to_categorical(
            y_train,
            num_classes=self.num_classes,
        )
        self.y_test = to_categorical(
            y_test,
            num_classes=self.num_classes,
        )

    def get_train_set(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train

    def get_test_set(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test


if __name__ == "__main__":
    m = MNIST()
    print(m)
