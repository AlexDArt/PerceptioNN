import os.path
import numpy as np
import tensorflow as tf

from keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop


# Создание модели нейронной сети
def create_model():
    m = tf.keras.models.Sequential([
        tf.keras.layers.Input(4),
        tf.keras.layers.Dense(3, activation='tahn'),
        tf.keras.layers.Dense(2, activation='tahn'),
        tf.keras.layers.Dense(1, activation='tahn'),
    ])
    m.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mse', 'mae', 'mape'])

    m.save('./models/model.h5')
    return m


# Проверка существования модели
if os.path.isfile('./models/model.h5'):
    model = tf.keras.models.load_model('./models/model.h5')
else:
    model = create_model()
# Обучающая выборка на вход
x_train = np.array([
    [.751, .005, .821, .02], [.738, .033, .938, .02],
    [.735, .016, .583, .03], [.723, .016, .951, .01],
    [.744, .06, .739, .01], [.760, -0.009, .315, .02],
    [.738, .103, .319, .02], [.739, .152, .184, .02]
])
# Обучающая выборка на выход
y_train = np.array([[.0861], [.0664], [.0810], [.0716], [.0808], [.0672], [.1189], [.1137]])
# Обучение модели с помоью обучающих выборок в 5 эпох
model.fit(x_train, y_train, epochs=1000)
# Сохранение обученной модели нейронной сети
model.save('./models/model.h5')
# Тестовые входные данные
x_test = np.array([
    [.742, .044, .856, .01],
])
# Получение результата работы нейронной сети
predictions = model.predict(x_test)
# Вывод результатов нейронной сети
for i in predictions:
    print(norm(i))
