import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectPercentile, SelectFromModel, chi2, f_classif, mutual_info_classif
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns

# 数据读取
def data_input(file):
    data_raw = pd.read_table(file, index_col=0, low_memory=False)
    enc = LabelEncoder()
    data_raw['target'] = enc.fit_transform(data_raw['target'].values)
    
    train_set, test_set = train_test_split(data_raw, test_size = 0.2)
    return train_set, test_set

# 数据预处理
def data_prec(train_set, test_set):
    scaler = MinMaxScaler()
    features_train = train_set.drop(['target'], axis = 1)
    features_test = test_set.drop(['target'],axis = 1)
    labels_train = train_set.target
    labels_test = test_set.target
    features_test, features_valid, labels_test, labels_valid = train_test_split(features_test, labels_test, test_size=0.5)
    
    scaler.fit(features_train)
    features_train = scaler.transform(features_train)
    features_valid = scaler.transform(features_valid)
    features_test = scaler.transform(features_test)
    
    return features_train, labels_train, features_valid, labels_valid, features_test, labels_test

# 基准模型
clf_names = ["Nearest Neighbors", "SVM", "Decision Tree", "Random Forest", 
         "AdaBoost", "Naive Bayes", "Gradient Tree Boosing", "Logistic", "Lasso"]

classifiers = [
    KNeighborsClassifier(n_neighbors=20,weights='distance', n_jobs=-1), # 计算cpu数目，-1代表全部cpu
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=400, n_jobs=-1),
    AdaBoostClassifier(),
    GaussianNB(),
    GradientBoostingClassifier(),
    LogisticRegression(n_jobs=-1),
    LogisticRegression(solver = "saga",penalty = "l1", n_jobs = -1)
]

# 建模主函数
def model_report(train_features, train_labels, valid_features, valid_labels, test_features, test_labels,k_times=1):
    print()
    
    multi_res= []
    for k in range(k_times):
        for name, clf in zip(clf_names, classifiers):
            clf.fit(train_features, train_labels)
            cv_acc = cross_val_score(clf, train_features, train_labels, cv = 10).mean()
            cv_f1 = cross_val_score(clf, train_features, train_labels, cv = 10, scoring='f1').mean()
            cv_auc = cross_val_score(clf, train_features, train_labels, cv = 10, scoring='roc_auc').mean()
            valid_acc = clf.score(valid_features, valid_labels)
            test_acc = clf.score(test_features, test_labels)
            res_values = [cv_acc, cv_f1, cv_auc, valid_acc, test_acc]
            multi_res += res_values
    
    res_df = np.array(multi_res).reshape(k_times, 5, len(clf_names))
    res_mean = res_df.mean(axis=0)
    res_mean = pd.DataFrame(res_mean)
    res_mean.index = ('Train_ACC', 'Train_F1', 'Train_AUC', 'Valid_ACC', 'Test_ACC')
    res_mean.columns = clf_names
    
    res_std = res_df.std(axis=0)
    res_std = pd.DataFrame(res_std)
    res_std.index = ('Train_ACC', 'Train_F1', 'Train_AUC', 'Valid_ACC', 'Test_ACC')
    res_std.columns = clf_names
    
    return res_mean, res_std

# 特征选择函数
def features_select(train_data, valid_data, method):
    if method in ['T test', 'Mann-Whitney', 'Wilcox test']:
        disease_group = train_data[train_data.target == 0]
        norm_group = train_data[train_data.target == 1]
        features_index = []

        for i in range(0, train_data.shape[1]-1):
            if method == 'T test':
                fs = stats.ttest_ind(disease_group.iloc[:,i], norm_group.iloc[:,i])
            elif method == 'Mann-Whitney':
                if len(set(disease_group.iloc[:,i])) & len(set(norm_group.iloc[:,i])) != 1:
                    fs = stats.mannwhitneyu(disease_group.iloc[:,i], norm_group.iloc[:,i])
            elif method == 'Wilcox test':
                fs = stats.ranksums(disease_group.iloc[:,i], norm_group.iloc[:,i])

            if fs.pvalue <= 0.05:
                features_index.append(i)
    
    elif method == 'Origin':
        return train_data, valid_data
    
    else:
        features = train_data.drop(['target'], axis=1)
        labels = train_data.target
        
        if method in ['Chi2', 'F-test', 'Mutual information']:
            if method == 'Chi2':
                sel = SelectPercentile(chi2, percentile=10)
            elif method == 'F-test':
                sel = SelectPercentile(f_classif)
            elif method == 'Mutual information':
                sel = SelectPercentile(mutual_info_classif, percentile=10)
            features_new = sel.fit_transform(features, labels)
            
        elif method in ['Logistics', 'LASSO', 'Random Forest']:
            if method == 'Logistics':
                clf = LogisticRegression(random_state=625)
            elif method == 'LASSO':
                clf = LogisticRegression(random_state=625, solver='saga', penalty='l1')
            elif method == 'Random Forest':
                clf = RandomForestClassifier(random_state=625)
            clf = clf.fit(features, labels)
            sel = SelectFromModel(clf, prefit=True)
            features_new = sel.transform(features)
        
        features_index = list(np.where(sel.get_support() == True)[0])

    features_index.append(train_data.shape[1]-1)
    train_fs = train_data.iloc[:,features_index]
    valid_fs = valid_data.iloc[:,features_index]
          
    return train_fs, valid_fs

# Heatmap
def heatmap(data, plotname):
    plt.close()
    plot = sns.heatmap(data, annot=True, cmap='RdBu_r')
    plt.tight_layout()
    plt.savefig('{}_DEMO.png'.format(plotname), dpi=600, format='PNG')
    plt.show()

# 数据读取 & 特征选择
train_data, test_data = data_input('otu_table_L6_model.txt')

sel_names = ["Origin","T test", "Wilcox test", "Mann-Whitney","Chi2", "F-test", "Mutual information","Logistics","LASSO",
             "Random Forest"]
features_trains, labels_trains, features_valids, labels_valids, features_tests, labels_tests = [],[],[],[],[],[]

for sel in sel_names:
    train_fs, valid_fs = features_select(train_data, test_data, sel)
    features_train, labels_train, features_valid, labels_valid, features_test, labels_test = data_prec(train_fs, valid_fs)
    
    features_trains.append(features_train)
    labels_trains.append(labels_train)
    features_valids.append(features_valid)
    labels_valids.append(labels_valid)
    features_tests.append(features_test)
    labels_tests.append(labels_test)

import time
time_start=time.time()
print("AGP Demo:")

res_mean_list = []
res_std_list = []

for sel_name, X_train, y_train, X_test, y_test, X_valid, y_valid in zip(sel_names, features_trains, labels_trains,
                                                                       features_tests, labels_tests, features_valids,
                                                                       labels_valids):
    print("Feature selection method:%s" %sel_name)
    res_mean, res_std = model_report(X_train, y_train, X_valid, y_valid, X_test, y_test, k_times=2)
    res_mean_list.append(res_mean)
    res_std_list.append(res_std)
    
    print(res_mean)
    print("-----------------------------------------------------------------------------------------\n")

time_end=time.time()
print('totally cost',time_end-time_start)

def plot_heatmap(res_list, sel_names):
    train_auc, valid_acc, test_acc = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for res in res_list:
        train_auc = pd.concat((train_auc, res.reindex(['Train_AUC'])), axis = 0)
        valid_acc = pd.concat((valid_acc, res.reindex(['Valid_ACC'])), axis = 0)
        test_acc = pd.concat((test_acc, res.reindex(['Test_ACC'])), axis = 0)

    train_auc.index, valid_acc.index, test_acc.index = sel_names, sel_names, sel_names
    
    heatmap(train_auc, 'train_acc')
    heatmap(valid_acc, 'valid_Acc')
    heatmap(test_acc, 'test_Acc')

plot_heatmap(res_mean_list, sel_names)

#调整参数
#超参数空间:定义需要调整参数的值，具体可参考scikit网站每种模型的具体有那些。简要原则如下：
#随机森林：决策树个数（一般越大越好）、每个决策数使用的最大特征数
#梯度提升回归树：决策树个数（配合学习率，不是越大越好）、学习率
#核支持向量机：核、gamma、C
#k最近邻：邻居的个数
#ada模型：和梯度提升回归树一样

param_grids = {'n_estimators': (50, 100, 1000, 10000, 100000),
                'learning_rate': np.arange(0.1, 1, 0.1)
                }

#网格搜索
train_data, valid_data = data_input("../otu_table_L6_ALL_0102_model.txt")
train_data_rf, valid_data_rf = selP(train_data, valid_data, chi2,10)
X_train_rf, y_train_rf, X_test_rf, y_test_rf, X_valid_rf, y_valid_rf = data_pre(train_data_rf, valid_data_rf)
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score), 'F1-score': make_scorer(f1_score)}
clf = GradientBoostingClassifier(random_state = 2019)
gs = GridSearchCV(clf, param_grid=param_grids, scoring=scoring,
                         refit = "AUC", cv = 5, return_train_score=True)
gs.fit(X_train_rf, y_train_rf)
print(gs.cv_results_)
print(gs.best_score_)
print(gs.best_params_)
