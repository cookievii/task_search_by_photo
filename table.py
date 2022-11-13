import csv
import pathlib

import numpy as np
import tensorflow as tf

from settings import IMG_HEIGHT, IMG_WEDTH, FILE_NAME, CATEGORY, PREDICTIONS_VERBOSE


class Table:
    """
    Получает обученный объект MachineLearning.
    Заполняет и создает таблицу формата csv.
    """

    def __init__(self, model_ml):
        self.ml = model_ml

    @staticmethod
    def save(name, sorted_images: list):
        """Создает таблицу из полученного списка."""
        with open(f"{name}_sorted.csv", "w", encoding="utf-8") as f:
            writer = csv.writer(
                f, quoting=csv.QUOTE_NONNUMERIC, delimiter=",", lineterminator="\r"
            )
            writer.writerows(sorted_images)

    def make_sorting(self, path: pathlib.Path) -> list:
        """Сортирует снимки по категориям."""
        table = [[FILE_NAME, CATEGORY]]
        images = list(path.glob("*/*.png"))
        for elem in images:
            abs_path_to_file = pathlib.Path.absolute(elem)
            img = tf.keras.utils.load_img(abs_path_to_file, target_size=(IMG_HEIGHT, IMG_WEDTH))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            predictions = self.ml.model.predict(img_array, verbose=PREDICTIONS_VERBOSE)
            score = tf.nn.softmax(predictions[0])
            table.append([elem.name, self.ml.train_ds.class_names[np.argmax(score)]])
        return table

    def get_csv_with_sorted_image(self, path: pathlib.Path) -> None:
        table = self.make_sorting(path)
        self.save(path, table)
