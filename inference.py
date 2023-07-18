import numpy as np
from catboost import CatBoost


# Чтение названий колонок из файла
with open('column_names.txt', 'r') as file:
    column_names = file.read().splitlines()


# Путь к модели CatBoost
model_path = 'catboost_model.cbm'
# Переменная skills (пример)
skills = ['Python', 'SQL', 'PyTorch', 'NLP', 'Git', 'Docker','Linux', 'C++']
# Ввод переменных
variable_1 = 'fullDay'
variable_2 = 'between1And3'
variable_3 = 'full'
variable_4 = 'Москва'

# Создание вектора для хранения результатов
skills_vector = []

# Проверка наличия названий колонок в массиве skills
for column in column_names:
    if column in skills:
        skills_vector.append(1)
    else:
        skills_vector.append(0)


# Маппинг переменных к соответствующим числовым значениям
variable_1_mapping = {'flexible': 0, 'flyInFlyOut': 1, 'fullDay': 2, 'remote': 3, 'shift': 4}
variable_2_mapping = {'between1And3': 0, 'between3And6': 1, 'moreThan6': 2, 'noExperience': 3}
variable_3_mapping = {'full': 0, 'part': 1, 'probation': 2, 'project': 3}
variable_4_mapping = {'Калининградская область': 0, 'Ленинградская область': 1, 'Москва': 2, 'Московская область': 3,
                      'Новосибирская область': 4, 'Пензенская область': 5, 'Псковская область': 6,
                      'Республика Башкортостан': 7, 'Республика Татарстан': 8, 'Санкт-Петербург': 9,
                      'Томская область': 10, 'Тюменская область': 11}

# Преобразование переменных в числовые значения
variable_1_encoded = variable_1_mapping.get(variable_1)
variable_2_encoded = variable_2_mapping.get(variable_2)
variable_3_encoded = variable_3_mapping.get(variable_3)
variable_4_encoded = variable_4_mapping.get(variable_4)

# Создание вектора переменных
variables_vector = [variable_1_encoded, variable_2_encoded, variable_3_encoded, variable_4_encoded]
res_vector = variables_vector + skills_vector

# Загрузка модели CatBoost
model_res = CatBoost()
model_res.load_model(model_path)

# Прогнозирование на основе вектора переменных
prediction = model_res.predict(res_vector)

# Вывод прогнозируемого значения
print("Прогнозируемое значение:", prediction)
