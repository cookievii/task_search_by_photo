import pathlib

from machine_lerning import MachineLearning
from table import Table


def path_exists(path: str) -> pathlib.Path:
    """
    Проверяет правильность указанной пути:
        -если правильно возвращает путь,
        -если неправильно получаем исключение.
    """
    path = pathlib.Path(path)
    if pathlib.Path.exists(path):
        return path
    raise FileExistsError("Указанной пути не существует.")


if __name__ == "__main__":
    """
    Структура указываемой директории должна быть: ./dir/*(folder)/*.png:
        path_to_dataset - Для обучения указываем директорию датасета;
        path_to_testing_images - Можем указать любую другую папку для сортировки и создания таблицы,
            Пример: Папка "example" - содержит файлы из интернета.
    На выходе получаем отсортированный csv файл отсортированный по классам снимки
    Созданный csv файл будет заканчиваться на "_sorted.csv".
    """

    path_to_dataset = "dataset"
    model = MachineLearning().get_trained_model(path=path_exists(path_to_dataset))

    path_to_testing_images = "example"
    Table(model).get_csv_with_sorted_image(path=path_exists(path_to_dataset))  # Можно передать path_to_testing_images
