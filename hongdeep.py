# 라이브러리 import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

# csv 파일 불러오기
data = pd.read_csv('data.csv')

# 훈련데이터 생성하기
X_train = data.drop(['sum'], axis=1)
y_train = data['sum']

# 최적화 함수 불러오기, learning_rate : 학습률
rmsprop = RMSprop(learning_rate=0.01)

# 모델 생성하기
model = Sequential()
model.add(Dense(1,input_shape=(2,)))
model.compile(loss='mse',optimizer=rmsprop)

# 모델 학습 후 기록 저장
history = model.fit(X_train, y_train, epochs=300, verbose=False)

# 오차가 줄어드는 모습을 출력
X = history.epoch
y = history.history['loss']

plt.plot(X, y, label='Model Loss')
plt.legend()

# 실제 값과 모델의 예측 값 비교하기
x1 = y_train
y1 = model.predict(X_train)

for i, j in zip(x1, y1):
    print("실제 값: ", i, " 예측 값: ", j)