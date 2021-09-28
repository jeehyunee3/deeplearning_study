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
import numpy as np
import os
import tensorflow as tf

#생략 없이 출력
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_seq_items', None)

# seed 값 설정
seed = 3
np.random.seed(seed)
tf.random.set_seed(seed)

# 데이터 입력
train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv', sep=',')
test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv', sep=',')
label = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv', sep=',')

# 데이터 정보 확인
# print(train.head(5))
# print(train.info())

# 오브젝트 idx
idx = ['bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',
       'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']
# idx=['bin_4','nom_1','nom_4','nom_7','nom_8','nom_9','ord_2','ord_3','ord_5']
"""
#히트맵
colormap = plt.cm.gist_heat
plt.figure(figsize=(25,25))
sns.heatmap(df.corr(),linewidths=0.1,vmax=0.5, cmap=colormap,\
            linecolor='white', annot=True)
plt.show()


#그래프 히스토그램
for i in idx:
    grid = sns.FacetGrid(train, col='target')
    grid.map(plt.hist, i,  bins=10)
    plt.show()
"""

# 공백 바로 위의 데이터를 입력
train = train.fillna(method='ffill')
test = test.fillna(method='ffill')

# 데이터 삭제
# 'bin_3','nom_0','nom_2','nom_3','nom_5','nom_6','ord_1','ord_4'
train = train.drop(['id'], axis=1)
test = test.drop(['id'], axis=1)

# object 속성 숫자로 바꿔주기
e = LabelEncoder()
for i in idx:
    e.fit(train[i])
    train[i] = e.transform(train[i])
    e.fit(test[i])
    test[i] = e.transform(test[i])

# 훈련_데이터분류
trian_dataset = train.values
X_train = trian_dataset[:, 0:23]
Y_train = trian_dataset[:, 23]

# 샘플_데이터
test_dataset = test.values
X_sample = test_dataset[:, 0:23]

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=seed)

# 모델생성
model = Sequential()
model.add(Dense(79, input_dim=23, activation='relu'))
model.add(Dense(67, activation='relu'))
model.add(Dense(58, activation='relu'))
model.add(Dense(49, activation='relu'))
model.add(Dense(37, activation='relu'))
model.add(Dense(29, activation='relu'))
model.add(Dense(17, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 모델 저장 폴더 설정
MODEL_DIR = '../model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

# 모델 저장 조건 설정
modelpath = "../model/claim-{epoch:02d}-accuracy:{accuracy:.4f}.hdf5"

# 모델 업데이트 및 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='accuracy', verbose=1, save_best_only=True)

# 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor='accuracy', patience=50)

# 모델 실행 및 저장
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=2000, batch_size=500, verbose=0,
                    callbacks=[early_stopping_callback, checkpointer])
"""
# 테스트셋 실험 결과의 값을 저장
y_vloss=history.history['val_loss']
y_vacc=history.history['val_accuracy']

# 학습 셋 측정한 값을 저장
y_acc=history.history['accuracy']
y_loss=history.history['loss']

# x값을 지정하고 나머지 표시
x_len = np.arange(len(y_loss))

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

print('------------------------------------------------------------------------')
print("\n train_Accuracy: %.4f" % (model.evaluate(X_train, Y_train)[1]))
print("\n test_Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

# 예측 값과 실제 값의 csv저장
Y_prediction = model.predict(X_sample).flatten()
Y_prediction = np.round(Y_prediction, 1)

# 제출파일 만들기
submission = pd.DataFrame({
    "id": label["id"],
    "target": Y_prediction
})
# print(submission.head(5))
submission.to_csv('./submission.csv', index=False)