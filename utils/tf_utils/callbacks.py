from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import tensorflow as tf
from keras import Model
from keras.callbacks import Callback
from keras.callbacks import TensorBoard

from .plotting import plot_confusion_matrix
from .plotting import plot_to_image


if TYPE_CHECKING:
    from collections.abc import Callable


class ImageCallback(Callback):
    """Custom tensorboard callback, to store images."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        model: Model,
        x_test: np.ndarray,
        y_test: np.ndarray,
        log_dir: str = "./",
        classes_list: list | None = None,
        figure_fn: Callable | None = None,
        figure_title: str | None = None,
    ) -> None:
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        if classes_list is None:
            self.classes_list = list(range(self.y_test[0]))
        else:
            self.classes_list = classes_list
        self.log_dir = log_dir
        img_file = os.path.join(self.log_dir, "images")
        self.file_writer_images = tf.summary.create_file_writer(img_file)
        self.figure_fn = figure_fn
        if figure_title is None:
            self.figure_title = str(self.figure_fn)
        else:
            self.figure_title = figure_title

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict | None = None,  # noqa: ARG002
    ) -> None:
        y_pred_prob: np.ndarray = self.model(self.x_test, training=False)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(self.y_test, axis=1)

        if self.figure_fn:
            fig = self.figure_fn(y_pred, y_true, self.classes_list)
            tf_image = plot_to_image(fig)
            figure_title_curr_epoch = self.figure_title + str(epoch)
            with self.file_writer_images.as_default():
                tf.summary.image(figure_title_curr_epoch, tf_image, step=epoch)


class ConfusionMatrix(ImageCallback):
    """Custom tensorbard callback, to store the confusion matrix figure."""

    def __init__(
        self,
        model: Model,
        x_test: np.ndarray,
        y_test: np.ndarray,
        classes_list: list,
        log_dir: str,
    ) -> None:
        self.figure_fn = plot_confusion_matrix
        self.figure_title = "Confusion Matrix"
        super().__init__(
            model,
            x_test,
            y_test,
            log_dir=log_dir,
            classes_list=classes_list,
            figure_fn=self.figure_fn,
            figure_title=self.figure_title,
        )

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict | None = None,
    ) -> None:
        super().on_epoch_end(epoch, logs)


def schedule_fn(
    epoch: int,
) -> float:
    learning_rate = 1e-3
    if epoch < 5:  # noqa: PLR2004
        learning_rate = 1e-3
    elif epoch < 20:  # noqa: PLR2004
        learning_rate = 5e-4
    else:
        learning_rate = 1e-4
    return learning_rate


def schedule_fn2(
    epoch: int,
) -> float:
    if epoch < 10:  # noqa: PLR2004
        return 1e-3
    return float(1e-3 * np.exp(0.1 * (10 - epoch)))


def schedule_fn3(
    epoch: int,
) -> float:
    return float(1e-3 * np.exp(0.1 * (10 - epoch)))


def schedule_fn4(
    epoch: int,
) -> float:
    return float(1e-3 * np.exp(0.05 * (10 - epoch)))


def schedule_fn5(
    epoch: int,
) -> float:
    learning_rate = 1e-3
    if epoch < 100:  # noqa: PLR2004
        learning_rate = 1e-3
    elif epoch < 200:  # noqa: PLR2004
        learning_rate = 5e-4
    elif epoch < 300:  # noqa: PLR2004
        learning_rate = 1e-4
    else:
        learning_rate = 5e-5
    return learning_rate


class LRTensorBoard(TensorBoard):
    def __init__(
        self,
        log_dir: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict,
    ) -> None:
        logs.update(
            {"learning_rate": self.model.optimizer.learning_rate.numpy()},
        )
        super().on_epoch_end(epoch, logs)


if __name__ == "__main__":
    lrtb = LRTensorBoard("")
