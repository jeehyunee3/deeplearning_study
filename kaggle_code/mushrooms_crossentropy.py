#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# seed설정
seed = 0
np.random.seed(3)
tf.random.set_seed(3)

#인덱스명 리스트
idx=['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color',
     'stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring',
     'veil-color','ring-number','ring-type','spore-print-color','population','habitat']

# 데이터 입력
df = pd.read_csv('../dataset/mushrooms.csv',sep=',')

#데이터채우기
df.fillna(0)

#데이터 숫자로 변경
e = LabelEncoder()
e.fit(df['class'])
df['class'] = e.transform(df['class'])
for i in idx:
    e.fit(df[i])
    df[i] = e.transform(df[i])
    
#데이터 삭제
df=df.drop(['veil-type' ],axis=1)

# 데이터 분류
dataset = df.values
X = dataset[:,1:13].astype(float)
Y = dataset[:,0]

#훈련데이터, 테스트데이터 나누기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=seed)

# 모델을 설정합니다.
model = Sequential()
model.add(Dense(35, input_dim=12, activation='relu'))
model.add(Dense(23, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델을 컴파일합니다.
model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

# 모델을 실행합니다.
model.fit(X_train, Y_train, epochs=150, batch_size=150)

# 결과를 출력합니다.
print("\n Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))


