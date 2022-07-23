# bnn_language_models


Для запуска

1) Настриваем переменные окружения 

* KMP_DUPLICATE_LIB_OK = True (без этого PyCharm не работал на M1)
* WANDB_CONFIG_DIR = путь к сохранению логов/конфигов wandb(у меня просто /tmp)
* WANDB_KEY = API ключ WANDB
* PROJECT_PATH = ваш путь к проекту


2) Настройка конфигов в src/config.yaml

3) Запуск скрипта обучения python src/trainer.py

4) Логи трекать через wandb url
