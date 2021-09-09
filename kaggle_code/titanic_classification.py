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

# seed 값 설정
seed = 3
numpy.random.seed(seed)
tf.random.set_seed(seed)

# 데이터 입력
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv',sep=',')
label = pd.read_csv('../input/gender_submission.csv',sep=',')
test.insert(0,'Survived',label['Survived'])

    
#데이터 삭제
train_data=train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],axis=1)
test_data=test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],axis=1)
train_data=train_data.dropna(axis=0)
test_data=test_data.fillna(0)
#print(train_data.info())
#print(test_data.info())

#문자열을 숫자로 변환 후 원핫인코딩
trian_dataset = pd.get_dummies(train_data).values
test_dataset = pd.get_dummies(test_data).values

#데이터분류
X_train = trian_dataset[:,1:11]
Y_train = trian_dataset[:,0]
X_test = test_dataset[:,1:11]
Y_test = test_dataset[:,0]

#모델생성
model = Sequential()
model.add(Dense(13, input_dim=10, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=500, batch_size=10, verbose=1)


# 예측 값과 실제 값의 csv저장
Y_prediction = model.predict(X_test).flatten()
Y_prediction=numpy.round(Y_prediction)
Y_prediction=map(int, Y_prediction)

submission=pd.DataFrame({\
"PassengerId":label["PassengerId"],\
"Survived":Y_prediction\
})
print(submission)
submission.to_csv('hyuneekk_submission.csv', index=False)
 
"""
# 모델 저장 폴더 설정
MODEL_DIR = '../model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

# 모델 저장 조건 설정
modelpath="../model/titanic-{epoch:02d}-accuracy:{accuracy:.4f}.hdf5"

# 모델 업데이트 및 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='accuracy', verbose=1, save_best_only=True)

# 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor='accuracy', patience=50)

# 모델 실행 및 저장 #verbose=0, callbacks=[early_stopping_callback,checkpointer]
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1000, batch_size=10, verbose=0, 
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
"""
