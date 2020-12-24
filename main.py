import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('./datasets/train-fenlei.csv')
test = pd.read_csv('./datasets/test-feilei.csv')

# 显示所有列
pd.set_option('display.max_columns', None)

#显示有无缺失值
print(train.isnull().any())

#显示数据类型
print(train.info())

# 目标变量正负样本的分布即0与1的比例
p = train['CLASS'].value_counts()
plt.figure(figsize=(10, 6))
patches, l_text, p_text = plt.pie(p, labels=[0, 1], autopct='%1.2f%%', explode=(0, 0.1))
for t in p_text:
    t.set_size(15)
for t in l_text:
    t.set_size(15)
plt.show()

train_feature = train.drop(['ID','CLASS'], axis=1).values
train_lable = train['CLASS'].values
test_feature = test.drop(['ID'], axis=1).values
test_ID=test['ID'].values

from sklearn import decomposition
#获得数据，X为特征值，y为标记值
X=train_feature
y=train_lable

pca=decomposition.PCA()
pca.fit(X,y)
ratio=pca.explained_variance_ratio_
print("pca.components_",pca.components_.shape)
print("pca_var_ratio",pca.explained_variance_ratio_.shape)
#绘制图形
plt.plot([i for i in range(X.shape[1])],
    [np.sum(ratio[:i+1]) for i in range(X.shape[1])])
plt.xticks(np.arange(X.shape[1],step=5))
plt.yticks(np.arange(0,1.01,0.05))
plt.grid()
plt.show()

#数据降维#降低维度之后会下降预测率
pca=decomposition.PCA(n_components=100)
# newData=pca.fit_transform(train_feature)
# newTest=pca.fit_transform(test_feature)
newData=train_feature
newTest=test_feature

### 皮尔逊相关系数
nu_fea = newData.astype(int) # 选择数值类特征计算相关系数
nu_feb= pd.DataFrame(nu_fea)
nu_fec = nu_feb.sample(n=10, frac=None, replace=False, weights=None, random_state=None, axis=1)
print(nu_fec)
nu_fed = list(nu_fec)    # 特征名列表
pearson_mat = nu_fec[nu_fed].corr(method='pearson')    # 计算皮尔逊相关系数矩阵

plt.figure(figsize=(10,10))
sns.heatmap(pearson_mat, square=True, annot=True, cmap="YlGnBu")    # 用热度图表示相关系数矩阵
plt.show() # 展示热度图

#数据归一化使用StandardScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(newData)
train = sc.transform(newData)
test=sc.transform(newTest)
train=pd.DataFrame(train)
train_lable=pd.DataFrame(train_lable)


# K折交叉验证来评估模型
#这里结果为决策树模型更好
from sklearn.model_selection import KFold

def kFold_cv(X, y, classifier, **kwargs):

    kf = KFold(n_splits=10, shuffle=True)
    y_pred = np.zeros(len(y))  # 初始化y_pred数组

    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]  # 划分数据集
        clf = classifier(**kwargs)
        clf.fit(X_train, y_train)  # 模型训练
        y_pred[test_index] = clf.predict(X_test)  # 模型预测

    return y_pred

# 模型预测
from sklearn.linear_model import LogisticRegression as LR    # 逻辑回归
from sklearn.svm import SVC  # SVM
from sklearn.ensemble import RandomForestClassifier as RF    # 随机森林
from sklearn import tree #决策树

#将数据处理成可以放入kFold_cv函数中
X=train.iloc[:,0:100]
X=X.values
y=train_lable
y=y.values


#测试三个模型的优劣
lr_pred = kFold_cv(X, y, LR)
svc_pred = kFold_cv(X, y, SVC)
rf_pred = kFold_cv(X, y, RF)
#由于决策树模型初始化不同所以另写

kf = KFold(n_splits=10, shuffle=True, random_state=0)
tree_pred = np.zeros(len(y))    # 初始化y_pred数组
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
for train_index, test_index in kf.split(X):
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]    # 划分数据集
    clf.fit(X_train, y_train)    # 模型训练
    tree_pred[test_index] = clf.predict(X_test)    # 模型预测

from sklearn.metrics import precision_score, recall_score, f1_score
# 导入精确率、召回率、F1值等评价指标
scoreDf = pd.DataFrame(columns=['LR', 'SVM', 'RandomForest', 'Tree'])
pred = [lr_pred, svc_pred, rf_pred, tree_pred]
for i in range(4):
    r = recall_score(y, pred[i])
    p = precision_score(y, pred[i])
    f1 = f1_score(y, pred[i])
    scoreDf.iloc[:, i] = pd.Series([r, p, f1])

#给scoreDf加上索引
scoreDf.index = ['Recall', 'Precision', 'F1-score']
scoreDf=pd.DataFrame(scoreDf)
print(scoreDf.head())
sa=scoreDf.loc['Recall']
sa=sa.values
sb=scoreDf.loc['Precision']
sb=sb.values
sc=scoreDf.loc['F1-score']
sc=sc.values

labels = ['LR', 'SVC', 'RandomForest', 'Tree']
x = [i for i in range(len(labels))]
plt.figure()
plt.bar(x,sa, width=0.3, label="Recall")
plt.bar([i + 0.3 for i in x],sb, width=0.3, label="Precision")
plt.bar([i + 0.6 for i in x], sc, width=0.3, label="F1-score")
plt.xticks([i + 0.3 for i in x], labels)
plt.legend()
plt.show()


from sklearn import tree
kf = KFold(n_splits=10, shuffle=True, random_state=0)
y_pred = np.zeros(len(y))    # 初始化y_pred数组
# clf = tree.DecisionTreeClassifier( max_depth=3)
from sklearn.metrics import accuracy_score

# #部分参数调优
# #惩罚因子C参数调优
# #经过计算得到最好是5
# C_Value=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# Acccuracy=[]
# for i in range(20):
#     for train_index, test_index in kf.split(X):
#         clf = SVC(C=i+1)
#         X_train = X[train_index]
#         X_test = X[test_index]
#         y_train = y[train_index]    # 划分数据集
#         clf.fit(X_train, y_train)    # 模型训练
#         y_pred[test_index] = clf.predict(X_test)    # 模型预测
#         #y是真实的y_pred是预测的
#         acc = accuracy_score(y, y_pred)
#     acc = accuracy_score(y, y_pred)
#     Acccuracy.append(acc)
# plt.plot(C_Value,Acccuracy)
# plt.xlabel('C_Value')
# plt.ylabel('Acccuracy')
# plt.show()

# #部分参数调优
# #核函数参数调优
# #经过计算得到最好是rbf
# kernel_Value=['rbf','linear','poly','sigmoid']
# Acccuracy=[]
# for i in kernel_Value:
#     for train_index, test_index in kf.split(X):
#         clf = SVC(C=5,kernel=i)
#         X_train = X[train_index]
#         X_test = X[test_index]
#         y_train = y[train_index]    # 划分数据集
#         clf.fit(X_train, y_train)    # 模型训练
#         y_pred[test_index] = clf.predict(X_test)    # 模型预测
#         #y是真实的y_pred是预测的
#         acc = accuracy_score(y, y_pred)
#     acc = accuracy_score(y, y_pred)
#     Acccuracy.append(acc)
# plt.plot(kernel_Value,Acccuracy)
# plt.xlabel('kernel_Value')
# plt.ylabel('Acccuracy')
# plt.show()


# # 部分参数调优
# # gamma参数调优
# # 经过计算得到最好是0.03171
# gamma_Value=np.linspace(0, 0.1, 40)
# Acccuracy=[]
# for i in gamma_Value:
#     for train_index, test_index in kf.split(X):
#         clf = SVC(C=5,kernel='rbf',gamma=i+0.001)
#         X_train = X[train_index]
#         X_test = X[test_index]
#         y_train = y[train_index]    # 划分数据集
#         clf.fit(X_train, y_train)    # 模型训练
#         y_pred[test_index] = clf.predict(X_test)    # 模型预测
#         #y是真实的y_pred是预测的
#         acc = accuracy_score(y, y_pred)
#     acc = accuracy_score(y, y_pred)
#     Acccuracy.append(acc)
# plt.plot(gamma_Value,Acccuracy)
# plt.xlabel('gamma_Value')
# plt.ylabel('Acccuracy')
# plt.show()

# # 部分参数调优
# # shrinking参数调优
# #经过下面的十折交叉验证代码发现True和FALSE预测结果相同

# # 部分参数调优
# # probability参数调优
# #经过下面的十折交叉验证代码发现True和FALSE预测结果相同



#部分参数调优
#max_iter参数调优
#经过计算得到最好是-1
max_iter_value=[-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
Acccuracy=[]
for i in range(22):
    for train_index, test_index in kf.split(X):
        clf =SVC(C=5,kernel='rbf',gamma=0.03171,max_iter=i-1)
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]    # 划分数据集
        clf.fit(X_train, y_train)    # 模型训练
        y_pred[test_index] = clf.predict(X_test)    # 模型预测
        #y是真实的y_pred是预测的
        acc = accuracy_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    Acccuracy.append(acc)
plt.plot(max_iter_value,Acccuracy)
plt.xlabel('max_iter_value')
plt.ylabel('Acccuracy')
plt.show()

for i in range(10):
    for train_index, test_index in kf.split(X):
        # clf = tree.DecisionTreeClassifier(max_depth=2, min_impurity_decrease=0,min_samples_split=i+2)
        clf=SVC(C=5,kernel='rbf',gamma=0.03171,shrinking=True,probability=False,max_iter=-1)
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]    # 划分数据集
        clf.fit(X_train, y_train)    # 模型训练
        y_pred[test_index] = clf.predict(X_test)    # 模型预测
        #y是真实的y_pred是预测的
        acc = accuracy_score(y, y_pred)
    print("验证集准确率: {}".format(acc))

clf=SVC(C=5,kernel='rbf',gamma=0.03171,shrinking=True,probability=False,max_iter=-1)
clf.fit(train,train_lable)
test_label=clf.predict(test)
df = pd.DataFrame({'ID':test_ID,'CLASS':test_label})
df.to_csv("SVC.csv", index=False)

