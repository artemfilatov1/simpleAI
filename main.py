import keras as k
import numpy as np
import tensorflow as tf

input_data = np.array([0.1, 0.3, 0.4])
output_data = np.array([0.5, 0.7, 0.8])

model = k.Sequential()
model.add(k.layers.Dense(units=1, activation='linear'))
model.compile(loss="mse", optimizer="sgd")
fit_results = model.fit(x=input_data, y=output_data, epochs=1000)

predicted = model.predict([0.2])
print(predicted);
