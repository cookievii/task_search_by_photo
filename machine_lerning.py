import pathlib

import tensorflow as tf

from settings import (BATCH_SIZE, COUNT_EPOCHS, IMG_HEIGHT, IMG_WEDTH,
                      VALIDATION_SPLIT)


class MachineLearning:
    """Обучает различать снимки."""

    def __init__(self):
        self.train_ds = None
        self.val_ds = None
        self.model = None

    def load_dataset(self, dir_dataset: pathlib.Path) -> None:
        """Загружает датасет."""
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            dir_dataset,
            validation_split=VALIDATION_SPLIT,
            subset="training",
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WEDTH),
            batch_size=BATCH_SIZE,
        )

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            dir_dataset,
            validation_split=VALIDATION_SPLIT,
            subset="validation",
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WEDTH),
            batch_size=BATCH_SIZE,
        )

    def normalizer_train_ds(self) -> None:
        """Стандартизирует датасет."""
        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
        self.train_ds.map(lambda x, y: (normalization_layer(x), y))

    def make_autotune(self) -> None:
        """Включает буферацию."""
        self.train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self.val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    def create_model(self) -> None:
        """Создает модель обучения."""
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Rescaling(1.0 / 255),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(len(self.train_ds.class_names)),
            ]
        )

    def optimizer_model(self) -> None:
        """Оптимизирует модель."""
        self.model.compile(
            optimizer="adam",
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def training_model(self):
        """Обучает модель."""
        self.model.fit(self.train_ds, validation_data=self.val_ds, epochs=COUNT_EPOCHS)

    def get_trained_model(self, path: pathlib.Path):
        """Возвращает обученную модель."""
        self.load_dataset(path)
        self.normalizer_train_ds()
        self.make_autotune()
        self.create_model()
        self.optimizer_model()
        self.training_model()
        return self
