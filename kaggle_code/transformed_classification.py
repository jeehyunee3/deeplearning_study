#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy
import os
import tensorflow as tf

pd.set_option('display.max_columns',None)

# seed 값 설정
seed = 3
numpy.random.seed(seed)
tf.random.set_seed(seed)

#인덱스 리스트
idx=['Favorite Color','Favorite Music Genre', 'Favorite Beverage', 'Favorite Soft Drink']

# 데이터 입력
df = pd.read_csv('../dataset/transformed.csv')
df.sample(frac=1)
#데이터 정보 확인
#print(df.info())
#print(df.describe())

#그래프 히스토그램
#for i in idx:
#    grid = sns.FacetGrid(df, col='Gender')
#    grid.map(plt.hist, i,  bins=10)
#    plt.show()


#문자열을 숫자로 변환 후 원핫인코딩
dataset = pd.get_dummies(df).values

# 속성/클래스 데이터 분류
X = dataset[:,2:22]
Y = dataset[:,0:2]

#print(dataset[0:5,:])
#print(dataset[61:,:])
#print(dataset.shape)

#테스트셋, 학습셋 데이터 분류
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

# 모델 설정
model = Sequential()
model.add(Dense(15, input_dim=20, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

# 모델 저장 폴더 설정
MODEL_DIR = '../model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

# 모델 저장 조건 설정
modelpath="../model/transformed-{epoch:02d}-val_accuracy:{val_accuracy:.4f}.hdf5"

# 모델 업데이트 및 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_accuracy', verbose=1, save_best_only=True)

# 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=50)

# 모델 실행 및 저장 # 
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=150, batch_size=2, verbose=0, 
                    callbacks=[early_stopping_callback,checkpointer])

# 테스트셋 실험 결과의 값을 저장
y_vloss=history.history['val_loss']
y_vacc=history.history['val_accuracy']

# 학습 셋 측정한 값을 저장
y_acc=history.history['accuracy']
y_loss=history.history['loss']

# x값을 지정하고 나머지 표시
x_len = numpy.arange(len(y_loss))
#loss그래프
plt.plot(x_len, y_vloss, "o", c="red",  markersize=5, label='Testset')
plt.plot(x_len, y_loss, "o", c="blue", markersize=5, label='Trainset')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
#acc그래프
plt.plot(x_len, y_vacc, "o", c="pink", markersize=5, label='Testset')
plt.plot(x_len, y_acc, "o", c="skyblue", markersize=5, label='Trainset')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()
