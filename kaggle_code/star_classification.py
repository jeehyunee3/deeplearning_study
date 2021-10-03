#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os


# seed 값 설정
seed = 3
numpy.random.seed(seed)
tf.random.set_seed(seed)

#인덱스 리스트
idx=['Temperature','L','R','A_M','Color','Spectral_Class']

# 데이터 입력
df = pd.read_csv('/kaggle/input/star-type-classification/Stars.csv')

#데이터 정보 확인
#print(df.info())
#print(df.describe())

"""
#히트맵
colormap = plt.cm.gist_heat
plt.figure(figsize=(25,25))
sns.heatmap(df.corr(),linewidths=0.1,vmax=0.5, cmap=colormap, linecolor='white', annot=True)
plt.show()

#그래프 히스토그램
plt.rc('font', size=7)
for i in idx:
    grid = sns.FacetGrid(df, col='Type')
    grid.map(plt.hist, i,  bins=10)
    plt.xticks(rotation=45)
    plt.show()
"""

df_x=df.drop(['Type'],axis=1)
df_y=df['Type']

X=pd.get_dummies(df_x).values
Y=pd.get_dummies(df_y).values

print(X.shape)
print(Y.shape)

#테스트셋, 학습셋 데이터 분류
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

# 모델 설정
model = Sequential()
model.add(Dense(20, input_dim=28, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

# 모델 저장 폴더 설정
MODEL_DIR = '../model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

# 모델 저장 조건 설정
modelpath="../model/stars-{epoch:02d}-accuracy:{accuracy:.4f}.hdf5"

# 모델 업데이트 및 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='accuracy', verbose=1, save_best_only=True)

# 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor='accuracy', patience=50)

# 모델 실행 및 저장 # 
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=200, batch_size=5, verbose=0, 
                    callbacks=[early_stopping_callback,checkpointer])

print('-----------------------------------------------')
print("\n train_Accuracy: %.4f" % (model.evaluate(X_train, Y_train)[1]))
print("\n test_Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

"""
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
"""
