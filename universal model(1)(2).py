# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Dongsuo Yin
# time:2019-07-10

# 包导入
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.cluster import KMeans,DBSCAN
from sklearn.metrics import accuracy_score, recall_score, f1_score,auc,silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

# 设置pycharm显示列数
pd.set_option('display.max_columns',None)

# 数据读入
path = r'C:\Users\hsyds\Desktop\HR.csv'  # 替换为训练数据目录
# data = pd.read_excel(path) # excel文件
data = pd.read_csv(path) # csv文件

# 数据查看
data.head()   # 查看前5行数据
data.shape    # 查看数据行列数
data.info()   # 查看数据条数和类型
data.describe().T   # 查看数据个数，均值，最大最小值，四分位

# 数据清洗
data_columns = data.columns.tolist()  # 列名称
data = data.dropna() # 清除无效值行

# 数值化
for i in data_columns:
    if data[i].dtype == 'object':
        temp = data[i].unique()
        for inx,content in enumerate(temp):
            data[i].replace(content,inx,inplace=True)

# 数据切分
label = 'left'  #填写作为标签的列名
X = data.drop(labels=label,axis=1)
Y = data[label]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=7)

# 数据归一化
for i in X.columns.tolist():
    data[i] = MinMaxScaler().fit_transform(data[i].values.reshape(-1,1)).reshape(1,-1)[0]

# 分类模型
# 1.随机森林（多分类）
model_rfc = RandomForestClassifier()
n_estimators = [10,50,100,200,300]
criterion = ['gini','entropy']
max_depth = [10,20,30]
param_grid = dict(n_estimators=n_estimators,criterion=criterion,max_depth=max_depth)
kflod = StratifiedKFold(n_splits=5, shuffle=True,random_state=7)
grid_search = GridSearchCV(model_rfc,param_grid,scoring='f1_micro',n_jobs=-1,cv=kflod)
grid_result = grid_search.fit(X_train, Y_train) #运行网格搜索
print(grid_result.best_params_)
print(grid_result.best_score_)
RFC_best = grid_result.best_estimator_
joblib.dump(RFC_best,'RFC_M')

# 2.随机森林（二分类）
model_rfc = RandomForestClassifier()
n_estimators = [10,50,100,200,300]
criterion = ['gini','entropy']
max_depth = [10,20,30]
param_grid = dict(n_estimators=n_estimators,criterion=criterion,max_depth=max_depth)
kflod = StratifiedKFold(n_splits=5, shuffle=True,random_state=7)
grid_search = GridSearchCV(model_rfc,param_grid,scoring='f1',n_jobs=-1,cv=kflod)
grid_result = grid_search.fit(X_train, Y_train) #运行网格搜索
print(grid_result.best_params
_)
print(grid_result.best_score_)
RFC_best = grid_result.best_estimator_
joblib.dump(RFC_best,'RFC_B')

# 3.XGBoost（多分类）
model_xgb = XGBClassifier() # 多分类'multi:softmax'
learning_rate = [0.01,0.1,0.2,0.3] #学习率[0.0001,0.001,0.01,0.1,0.2,0.3]
gamma = [1, 0.1,0.01] # [1, 0.1, 0.01, 0.001]
max_depth = [4,5,6,7]
objective=['multi:softmax'] # 多分类
num_class=[len(Y.unique())] # 分类数目
param_grid = dict(learning_rate = learning_rate,gamma = gamma,max_depth=max_depth,objective=objective,num_class=num_class) #转化为字典格式，网络搜索要求
kflod = StratifiedKFold(n_splits=5, shuffle=True,random_state=7) #将训练/测试数据集划分5个互斥子集
grid_search = GridSearchCV(model_xgb,param_grid,scoring='f1_micro',n_jobs=-1,cv=kflod)
grid_result = grid_search.fit(X_train, Y_train) #运行网格搜索
print(grid_result.best_params_)
print(grid_result.best_score_)
XGB_best = grid_result.best_estimator_
joblib.dump(XGB_best,'XGB_M')

# 4.XGBoost（二分类）
model_xgb = XGBClassifier() # 多分类'multi:softmax'
learning_rate = [0.01,0.1,0.2,0.3] #学习率[0.0001,0.001,0.01,0.1,0.2,0.3]
gamma = [1, 0.1,0.01] # [1, 0.1, 0.01, 0.001]
max_depth = [4,5,6,7]
param_grid = dict(learning_rate = learning_rate,gamma = gamma,max_depth=max_depth) #转化为字典格式，网络搜索要求
kflod = StratifiedKFold(n_splits=5, shuffle=True,random_state=7) #将训练/测试数据集划分5个互斥子集
grid_search = GridSearchCV(model_xgb,param_grid,scoring='f1',n_jobs=-1,cv=kflod)
grid_result = grid_search.fit(X_train, Y_train) #运行网格搜索
print(grid_result.best_params_)
print(grid_result.best_score_)
XGB_best = grid_result.best_estimator_
joblib.dump(XGB_best,'XGB_B')

# 聚类模型
# DBSCAN
eps = [0.2,0.3,0.5,0.7] # 数据需要归一化
min_samples = [3,5,7]
# 以轮廓系数为指标，找寻最优参数
for i in eps:
    for j in min_samples:
        clustering = DBSCAN(eps=i,min_samples=j).fit(X)
        print('eps=%.2f,min_samples=%.2f'%(i,j))
        print(silhouette_score(X,clustering.labels_))
# 把最优参数填入模型，再次进行训练，并保存模型
model_dbs = DBSCAN(eps=0.7,min_samples=5).fit(X)
joblib.dump(model_dbs,'DBSCAN') # 保存模型

