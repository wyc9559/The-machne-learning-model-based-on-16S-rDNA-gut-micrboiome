#!/bin/env/python
#The selection progrom
#by Wu Tong in 23/12/2019
 
#载入必要的软件包 
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import VarianceThreshold,chi2, f_classif, mutual_info_classif,SelectPercentile,SelectFromModel,RFE
from sklearn.model_selection import cross_val_score, KFold,train_test_split, GridSearchCV
from sklearn.svm import SVC
from scipy import stats
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_curve, roc_auc_score, make_scorer, accuracy_score
from matplotlib import pyplot as pltipyt

#总模型数据读取函数
def data_input(file):
    data_raw = pd.read_csv(file, delimiter = "\t", index_col = 0)
    enc = LabelEncoder()
    data_raw['target'] = enc.fit_transform(data_raw['target'].values)
    train_data = data_raw[data_raw.index.str.match('(^[AES])')]
    valid_data = data_raw[data_raw.index.str.match('(^[FJ])')]
    return train_data, valid_data

#数据预处理函数
def data_pre(train_data,valid_data):
    scaler = MinMaxScaler()
    factor = train_data.drop(['target'],axis=1)
    factor_valid = valid_data.drop(['target'],axis =1)
    target = train_data.target
    target_valid = valid_data.target
    X_train,X_test,y_train,y_test = train_test_split(factor,target, random_state = 2019)
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    factor_valid_scaled = scaler.transform(factor_valid)
    return X_train_scaled,y_train,X_test_scaled,y_test,factor_valid_scaled,target_valid    

#基线模型
#mulit model
clf_names = ["Nearest Neighbors", "SVM", "Decision Tree", "Random Forest", 
         "AdaBoost", "Naive Bayes", "Gradient Tree Boosing", "Logistic", "Lasso"]

classifiers = [
    KNeighborsClassifier(3, n_jobs = 16),
    SVC(random_state = 2019),
    DecisionTreeClassifier(random_state = 2019),
    RandomForestClassifier(random_state = 2019, n_jobs = 16),
    AdaBoostClassifier(random_state = 2019),
    GaussianNB(),
    GradientBoostingClassifier(random_state = 2019),
    LogisticRegression(random_state = 2019, n_jobs = 16),
    LogisticRegression(random_state = 2019, solver = "saga",penalty = "l1", n_jobs = 16)
]

#建模主函数
def model_report(X_train, y_train, X_test, y_test, X_valid, y_valid):
    print()
    res_dict = {}
    for name, clf in zip(clf_names, classifiers):
        clf.fit(X_train, y_train)
        cv_acc = cross_val_score(clf, X_train, y_train, cv = 5).mean()
        cv_f1 = cross_val_score(clf, X_train, y_train, cv = 5, scoring = "f1").mean()
        cv_auc = cross_val_score(clf, X_train, y_train, cv = 5, scoring = "roc_auc").mean()
        test_acc = clf.score(X_test, y_test)
        valid_acc = clf.score(X_valid, y_valid)
        res_value = (cv_acc, cv_f1, cv_auc, test_acc, valid_acc)
        res_dict[name] = res_value
    res_df = pd.DataFrame(res_dict)
    res_df.index = ("train_acc", "train_f1", "train_auc", 
                    "test_acc", "valid_auc")
    return(res_df)

#特征选择函数
#T_test
def t_test(train_data,valid_data):
    dis_group = train_data[train_data.target == 0]
    nor_group = train_data[train_data.target == 1]
    fea_index = []
    for i in range(0,train_data.shape[1]-2):
        t = stats.ttest_ind(dis_group.iloc[:,i], nor_group.iloc[:,i])
        if t.pvalue <= 0.05:
            fea_index.append(i)
    fea_index.append(train_data.shape[1]-1)
    train_t = train_data.iloc[:,fea_index]
    valid_t = valid_data.iloc[:, fea_index]
    return train_t, valid_t

#Mann-Whitney
def mann_test(train_data,valid_data):
    dis_group = train_data[train_data.target == 0]
    nor_group = train_data[train_data.target == 1]
    fea_index = []
    for i in range(0,train_data.shape[1]-2):
        if len(set(dis_group.iloc[:,i])) & len(set(nor_group.iloc[:,i])) != 1:
            m = stats.mannwhitneyu(dis_group.iloc[:,i], nor_group.iloc[:,i])
            if m.pvalue <= 0.05:
                fea_index.append(i)
    fea_index.append(train_data.shape[1]-1)
    train_m = train_data.iloc[:,fea_index]
    valid_m = valid_data.iloc[:,fea_index]
    return train_m,valid_m

#Wilcxo_test
def wilcox_test(train_data,valid_data):
    dis_group = train_data[train_data.target == 0]
    nor_group = train_data[train_data.target == 1]
    fea_index = []
    for i in range(0,train_data.shape[1]-2):
        w = stats.ranksums(dis_group.iloc[:,i], nor_group.iloc[:,i])
        if w.pvalue <= 0.05:
            fea_index.append(i)
    fea_index.append(train_data.shape[1]-1)
    train_w = train_data.iloc[:,fea_index]
    valid_w = valid_data.iloc[:,fea_index]
    return train_w,valid_w

#chi2 analysis, F, mutual_info
def selP(train_data, valid_data, method, per):
    X = train_data.drop(["target"], axis = 1)
    y = train_data.target
    sel = SelectPercentile(method, percentile = per)
    factor_new = sel.fit_transform(X, y)
    index = list(np.where(sel.get_support() == True)[0])
    index.append(train_data.shape[1]-1)
    train_new = train_data.iloc[:,index]
    valid_new = valid_data.iloc[:,index]
    return train_new, valid_new
    

#select form model
def selM(train_data, valid_data, clf):
    X = train_data.drop(["target"], axis = 1)
    y = train_data.target
    clf = clf.fit(X,y)
    sel = SelectFromModel(clf, prefit = True)
    X_new = sel.transform(X)
    index = list(np.where(sel.get_support() == True)[0])
    index.append(train_data.shape[1]-1)
    train_new = train_data.iloc[:,index]
    valid_new = valid_data.iloc[:,index]
    return train_new, valid_new

#特征选择
train_data_t, valid_data_t = t_test(train_data, valid_data)
train_data_w, valid_data_w = wilcox_test(train_data, valid_data)
train_data_m, valid_data_m = mann_test(train_data, valid_data)
train_data_chi2, valid_data_chi2 = selP(train_data, valid_data, chi2, 10)
train_data_f, valid_data_f = selP(train_data, valid_data, f_classif, 10)
train_data_minfo, valid_data_minfo = selP(train_data, valid_data, mutual_info_classif, 10)
log = LogisticRegression(random_state = 2019)
train_data_log, valid_data_log = selM(train_data, valid_data, log)
lasso = LogisticRegression(random_state = 2019, solver = "saga", penalty = "l1")
train_data_lasso, valid_data_lasso = selM(train_data, valid_data, lasso)
rf = RandomForestClassifier(random_state = 2019)
train_data_rf, valid_data_rf = selM(train_data, valid_data, rf)

#特征数据集划分
X_train_t, y_train_t, X_test_t, y_test_t, X_valid_t, y_valid_t = data_pre(train_data_t, valid_data_t)
X_train_w, y_train_w, X_test_w, y_test_w, X_valid_w, y_valid_w = data_pre(train_data_w, valid_data_w)
X_train_m, y_train_m, X_test_m, y_test_m, X_valid_m, y_valid_m = data_pre(train_data_m, valid_data_m)
X_train_c, y_train_c, X_test_c, y_test_c, X_valid_c, y_valid_c = data_pre(train_data_chi2, valid_data_chi2)
X_train_f, y_train_f, X_test_f, y_test_f, X_valid_f, y_valid_f = data_pre(train_data_f, valid_data_f)
X_train_mi, y_train_mi, X_test_mi, y_test_mi, X_valid_mi, y_valid_mi = data_pre(train_data_minfo, valid_data_minfo)
X_train_lg, y_train_lg, X_test_lg, y_test_lg, X_valid_lg, y_valid_lg = data_pre(train_data_log, valid_data_log)
X_train_la, y_train_la, X_test_la, y_test_la, X_valid_la, y_valid_la = data_pre(train_data_lasso, valid_data_lasso)
X_train_rf, y_train_rf, X_test_rf, y_test_rf, X_valid_rf, y_valid_rf = data_pre(train_data_rf, valid_data_rf)

#所有数据集名称
sel_names = ["Origin","T test", "Wilcox test", 
             "Mann-Whitney","Chi2", "F-test", 
             "Mutual information","Logstiics","LASSO",
             "Random Forest"]
X_trains = [X_train_o, X_train_t, X_train_w, 
            X_train_m, X_train_c, X_train_f, 
            X_train_mi, X_train_lg, X_train_la,
           X_train_rf]
y_trains = [y_train_o, y_train_t, y_train_w,
           y_train_m, y_train_c, y_train_f,
           y_train_mi, y_train_lg, y_train_la,
           y_train_rf]
X_tests = [X_test_o, X_test_t, X_test_w,
          X_test_m, X_test_c, X_test_f,
          X_test_mi, X_test_lg, X_test_la,
          X_test_rf]
y_tests = [y_test_o, y_test_t, y_test_w,
          y_test_m, y_test_c, y_test_f,
          y_test_mi, y_test_lg, y_test_la,
          y_test_rf]
X_valids = [X_valid_o, X_valid_t, X_valid_w,
           X_valid_m, X_valid_c, X_valid_f,
           X_valid_mi, X_valid_lg, X_valid_la,
           X_valid_rf]
y_valids = [y_valid_o, y_valid_t, y_valid_w,
           y_valid_m, y_valid_c, y_valid_f,
           y_valid_mi, y_valid_lg, y_valid_la,
           y_valid_rf]
           
#构建模型
train_data, valid_data = data_input("./data/otu_L6_ALL_Model.txt")
X_train_o,y_train_o,X_test_o,y_test_o,X_valid_o,y_valid_o = data_pre(train_data,valid_data)

#建立模型过程
print("The Orgin OTU Table Model:")

for sel_name, X_train, y_train, X_test, y_test, X_valid, y_valid in zip(sel_names, X_trains, y_trains,
                                                                       X_tests, y_tests, X_valids,
                                                                       y_valids):
    print("Feature selection method:%s" %sel_name)
    res = model_report(X_train, y_train, X_test, y_test, X_valid, y_valid)
    print(res)
    print("-----------------------------------------------------------------------------------------\n")
