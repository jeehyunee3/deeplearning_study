#!/usr/bin/env python

# -*- coding: utf-8 -*-
# 코드 내부에 한글을 사용가능 하게 해주는 부분입니다.

# pandas 라이브러리를 불러옵니다.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 유리 데이터셋을 불러옵니다.
df = pd.read_csv('../dataset/glass.csv')

# 처음 5줄을 봅니다.
#print(df.head(5))

# 데이터의 전반적인 정보를 확인해 봅니다.
#print(df.info())

# 각 정보별 특징을 좀더 자세히 출력합니다.
#print(df.describe())

# 데이터 간의 상관관계를 그래프로 표현해 봅니다.
#colormap = plt.cm.gist_heat   #그래프의 색상 구성을 정합니다.
#plt.figure(figsize=(12,12))   #그래프의 크기를 정합니다.

# 그래프의 속성을 결정합니다. vmax의 값을 0.5로 지정해 0.5에 가까울 수록 밝은 색으로 표시되게 합니다.
#sns.heatmap(df.corr(),linewidths=0.1,vmax=0.5, cmap=colormap, linecolor='white', annot=True)
#plt.show()

#히스토그램
grid = sns.FacetGrid(df, col='Type')
grid.map(plt.hist, 'RI',  bins=10)
plt.show()

grid = sns.FacetGrid(df, col='Type')
grid.map(plt.hist, 'Na',  bins=10)
plt.show()

grid = sns.FacetGrid(df, col='Type')
grid.map(plt.hist, 'Mg',  bins=10)
plt.show()

grid = sns.FacetGrid(df, col='Type')
grid.map(plt.hist, 'Al',  bins=10)
plt.show()

grid = sns.FacetGrid(df, col='Type')
grid.map(plt.hist, 'Si',  bins=10)
plt.show()

grid = sns.FacetGrid(df, col='Type')
grid.map(plt.hist, 'K',  bins=10)
plt.show()

grid = sns.FacetGrid(df, col='Type')
grid.map(plt.hist, 'Ca',  bins=10)
plt.show()


grid = sns.FacetGrid(df, col='Type')
grid.map(plt.hist, 'Ba',  bins=10)
plt.show()

grid = sns.FacetGrid(df, col='Type')
grid.map(plt.hist, 'Fe',  bins=10)
plt.show()
