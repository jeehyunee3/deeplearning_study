# 필요한 패키지 import
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import zipfile

# seed 값 설정
seed = 3
np.random.seed(seed)
tf.random.set_seed(seed)

# 이미지 관련 변수 선언
image_width = 128
image_height = 128
image_size = (image_width, image_height)
image_channels = 3
batch_size = 250

# zip파일 경로 설정
data_zip_dir = '/kaggle/input/dogs-vs-cats'
train_zip_dir = os.path.join(data_zip_dir, 'train.zip')
test_zip_dir = os.path.join(data_zip_dir, 'test1.zip')

# 압축해제
with zipfile.ZipFile(train_zip_dir, 'r') as z:
    z.extractall()
with zipfile.ZipFile(test_zip_dir, 'r') as z:
    z.extractall()
train_dir = os.path.join(os.getcwd(), 'train')
test_dir = os.path.join(os.getcwd(), 'test')

# train데이터 불러오기 및 정답 설정
filenames = os.listdir(train_dir)
categories = []
for i in filenames:
    if 'dog' in i:
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
# print(df.head(5))
# print(df.tail(5))

# 정답데이터 str 변환
df['category'] = df['category'].replace({1: 'dog', 0: 'cat'})

# train, validate 분리
train_df, validate_df = train_test_split(df, test_size=0.3, random_state=seed)

# 컨볼루션 신경망의 설정
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# print(model.summary())

# 콜백함수 정의
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = "./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# train, validate 갯수 확인
train_size = train_df.shape[0]
validate_size = validate_df.shape[0]
# print(train_size, validate_size)

# train, validate 데이터 부풀리기
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1.0 / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1)

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    train_dir,
                                                    x_col='filename',
                                                    y_col='category',
                                                    target_size=image_size,
                                                    class_mode='categorical',
                                                    batch_size=batch_size)

validation_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1.0 / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1)

validation_generator = validation_datagen.flow_from_dataframe(validate_df,
                                                              train_dir,
                                                              x_col='filename',
                                                              y_col='category',
                                                              target_size=image_size,
                                                              class_mode='categorical',
                                                              batch_size=batch_size)

# 모델의 실행
history = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                              epochs=30,
                              validation_steps=validate_size // batch_size,
                              steps_per_epoch=train_size // batch_size,
                              callbacks=[early_stopping_callback, checkpointer])

# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Validateset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# test데이터 가져오기
test_filenames = os.listdir(test_zip_dir)
test_df = pd.DataFrame({
    'filename': test_filenames
})
test_size = test_df.shape[0]

# test 데이터 부풀리기
test_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1.0 / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1)

test_generator = test_datagen.flow_from_dataframe(test_df,
                                                  "./dataset/dogcat/test1/",
                                                  x_col='filename',
                                                  y_col=None,
                                                  target_size=image_size,
                                                  class_mode=None,
                                                  batch_size=batch_size,
                                                  shuffle=False)

# 모델예측
predict = model.predict_generator(test_generator, steps=np.ceil(test_size / batch_size))
test_df['category'] = np.argmax(predict, axis=-1)
test_df['category'] = test_df['category'].replace({'dog': 1, 'cat': 0})

for i in range(0, test_size):
    test_df['filename'][i] = int(test_df['filename'][i].replace('.jpg', ''))
    # print(test_df['filename'][i])

test_df = test_df.sort_values(ascending=True, by='filename')

test_df.to_csv("submission.csv", header=False, index=False)