#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)

# 데이터 입력
df = pd.read_csv('../dataset/glass.csv')
df['Type'].fillna(1)
df.fillna(0)

# 그래프로 확인
#sns.pairplot(df, hue='Type');
#plt.show()

# 데이터 분류
dataset = df.values
X = dataset[:,0:9].astype(float)
Y_obj = dataset[:,9]

# 문자열을 숫자로 변환
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = tf.keras.utils.to_categorical(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.3, random_state=seed)

# 모델의 설정
model = Sequential()
model.add(Dense(26,  input_dim=9, activation='relu'))
model.add(Dense(15,  activation='relu'))
model.add(Dense(9,  activation='relu'))
model.add(Dense(6, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

# 모델 실행
model.fit(X_train, Y_train, epochs=1000, batch_size=20)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))
