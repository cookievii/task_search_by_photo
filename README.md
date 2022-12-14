# Тестовое задание: Сортировка снимков с ML.

----------

### Стэк технологий:

* Python == 3.10
* tensorflow == 2.10.0
* keras == 2.10.0
* numpy == 1.23.4

# Описание проекта

* *Исходные данные для разработки:*

* * _Две папки с равным количеством снимков в негативе и позитиве (негатив - кости белые, позитив - кости черные)._


* *Формулировка задачи:*

* * _Входные данные для алгоритма – папка с цифровыми флюорограммами в формате png, в которой 50/50 негативных и 
позитивных снимков. Необходимо разработать алгоритм (Python скрипт) сортировки флюорограмм на 2 класса: негатив 
(n)/позитив (p)._


* * _Выходные данные – таблица csv файл со следующими столбцами: имя файла, результат сортировки 
(«n» или «p»). Оценка алгоритма будет выполняться на другой (валидационной) выборке._


## Установка проекта

```bash
# - Скачиваем проект.
git clone git@github.com:cookievii/task_search_by_photo.git

# - Переходим в директорию проекта 
cd task_search_by_photo

# - Cоздаем виртуальное окружение и активируем.
python -m venv venv
source venv/bin/activate

# - Устанавливаем зависимости из файла "requirements.txt".
pip install -r requirements.txt

# - Открываем main.py - проверяем пути и запускаем main.py если все устраивает.
python main.py
```

##### *Могут потребоваться дополнительные компоненты для ОС:*
1. Установка tensorflow: https://www.tensorflow.org/install

2. Если ошибка "Could not locate zlibwapi.dll" - Поможет статья: [stackoverflow](https://stackoverflow.com/questions/72356588/could-not-locate-zlibwapi-dll-please-make-sure-it-is-in-your-library-path)
