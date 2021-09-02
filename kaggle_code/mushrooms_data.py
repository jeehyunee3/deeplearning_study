#!/usr/bin/env python

# -*- coding: utf-8 -*-
# 코드 내부에 한글을 사용가능 하게 해주는 부분입니다.

# pandas 라이브러리를 불러옵니다.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../dataset/mushrooms.csv')

idx=['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape',
'stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring',
'veil-type', 'veil-color','ring-number','ring-type','spore-print-color','population','habitat']

#문자열 숫자로 변경
e = LabelEncoder()
e.fit(df['class'])
df['class'] = e.transform(df['class'])
for i in idx:
    e.fit(df[i])
    df[i] = e.transform(df[i])


colormap = plt.cm.gist_heat   #그래프의 색상 구성을 정합니다.
plt.figure(figsize=(25,25))   #그래프의 크기를 정합니다.

# 그래프 히트맵
sns.heatmap(df.corr(),linewidths=0.1,vmax=0.5, cmap=colormap, linecolor='white', annot=True)
plt.show()

#그래프 히스토그램
for i in temp:
    grid = sns.FacetGrid(df, col='class')
    grid.map(plt.hist, i,  bins=10)
    plt.show()

#데이터 관계확인
for i in idx:
    print(df[[i,'class']].groupby([i], as_index=False).mean().sort_values(by='class',ascending=False))

