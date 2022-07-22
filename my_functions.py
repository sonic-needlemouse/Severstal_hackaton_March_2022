import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder 
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, \
    mean_absolute_error, median_absolute_error, roc_auc_score, roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix
import itertools
import datetime
import tqdm
import re

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import *
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from pandas.plotting import scatter_matrix

import seaborn as sns


def rename_cols(df):
    df.columns = df.columns.str.replace(' ', '_', regex=True)
    df.columns = df.columns.str.replace('-', '_', regex=True)
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace('№', 'N', regex=True)
    chars_to_remove = ['.', ',', '(', ')', ' ', '/', ':', '[', ']']
    regular_expression = '[' + re.escape(''.join(chars_to_remove)) + ']'
    df.columns = df.columns.str.replace(regular_expression, '', regex=True)
    return df

def join_data(df_list):
    
    # Создание общего датафрейма на основе данных за 2021 г (наиболее полные, содержат в т.ч. данные за предыдущие годы)
    df_total = df_list[2].drop(columns=['Unnamed: 0'])
    df_total.index = df_total['Наименование ДП']
    rename_dict = {'Факт.32':'Факт 32',
                   'Факт.31':'Факт 31',
                'Факт.23':'Факт 23'}
    df_total = df_total.rename(columns=rename_dict)
    df_total = rename_cols(df_total)
    
    df_total['кол_во_раз_пдз_за_2021_год_шт'] = 0
    for idx in df_total.index:
        for col_name in ['пдз_1_30', 'пдз_31_90', 'пдз_91_365', 'пдз_более_365']:
            df_total.loc[idx,'кол_во_раз_пдз_за_2021_год_шт'] += df_total.loc[idx,col_name]
            
    # добавление года к колонкам, где его нет
    L = []
    for col_name in df_total.columns:
        k = 0
        for pattern in ['20{}'.format(i) for i in range(16,22)]:
            if pattern in col_name:
                k = 1
        if k == 0:
            L.append(col_name)
            
    rename_dict = {}
    for col_name in L:
        if col_name != 'наименование_дп':
            rename_dict[col_name] = '2021_'+col_name
    df_total = df_total.rename(columns=rename_dict)
    
    # Добавление данных за 2019 г (минимум фичей, но есть данные о новых дебиторах)
    df_tmp = df_list[0].drop(columns=['Unnamed: 0'])
    df_tmp.index = df_tmp['Наименование ДП']

    df_tmp = rename_cols(df_tmp)

    # добавление года к колонкам, где его нет
    L = []
    for col_name in df_tmp.columns:
        k = 0
        for pattern in ['20{}'.format(i) for i in range(16,22)]:
            if pattern in col_name:
                k = 1
        if k == 0:
            L.append(col_name)
            
    rename_dict = {}
    for col_name in L:
        if col_name != 'наименование_дп':
            rename_dict[col_name] = '2019_'+col_name
    df_tmp = df_tmp.rename(columns=rename_dict)

    add_idx_list=[]
    for idx in df_tmp.index:
        if idx not in df_total.index:
            add_idx_list.append(idx)

    print(len(add_idx_list))

    for col_name in df_tmp.columns:
        if col_name not in df_total.columns:
            print(col_name)
            
    df_total = pd.concat([df_total, df_tmp.loc[add_idx_list,:]])
    df_total = df_total.sort_index()
    
    # Добавление данных за 2020 г (нет новых дебиторов, но есть новые фичи по 2020 г - факт, итого и пр.)
    df_tmp = df_list[1].copy()
    df_tmp.index = df_tmp['Наименование ДП']

    df_tmp = rename_cols(df_tmp)

    # добавление года к колонкам, где его нет
    L = []
    for col_name in df_tmp.columns:
        k = 0
        for pattern in ['20{}'.format(i) for i in range(16,22)]:
            if pattern in col_name:
                k = 1
        if k == 0:
            L.append(col_name)
    # print(L)
    rename_dict = {}
    for col_name in L:
        if col_name != 'наименование_дп':
            rename_dict[col_name] = '2020_'+col_name
    df_tmp = df_tmp.rename(columns=rename_dict)

    extra_cols_2020 = []
    for col_name in df_tmp.columns:
        if col_name not in df_total.columns:
            extra_cols_2020.append(col_name)
            
    df_total = df_total.join(df_tmp[extra_cols_2020])
    
    return df_total
    
    
def prepare_data_for_model(df_total):
    df_by_years_list = []

    for year in [2021, 2020, 2019]:
        cols_list = ['наименование_дп']
        for col_name in df_total.columns:
            if str(year) in col_name or \
                str(year-1) in col_name or \
                str(year-2) in col_name or \
                str(year-3) in col_name or \
                str(year-4) in col_name or \
                str(year-5) in col_name:
                cols_list.append(col_name)
        df_tmp = df_total[cols_list]

        rename_dict = {}

        for i in range(6):
            for col_name in df_tmp.columns:
                if str(year-i) in col_name:
                    if i == 0:
                        rename_dict[col_name] = col_name.replace(str(year-i)+'_', '')
                    else:
                        rename_dict[col_name] = col_name.replace(str(year-i), 'YY_{}'.format(i))

        df_tmp = df_tmp.rename(columns=rename_dict)
        df_tmp['YY'] = year
        df_by_years_list.append(df_tmp)

    df_by_years_total = pd.concat([df_by_years_list[0], df_by_years_list[1], df_by_years_list[2]])
    df_by_years_total.index = range(len(df_by_years_total.index))
    
    return df_by_years_total
    
    
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(5, 3))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Факт')
    plt.xlabel('Прогноз\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def plot_feature_importance(importance,names,x,y):
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    plt.figure(figsize=(x,y))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title('FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    return data

def plot_target_prediction(y_test, y_pred, coord_start, coord_end):
    plt.figure(figsize=(10,10))
    plt.scatter(y_test, y_pred, s=10, alpha=0.7)
    plt.grid(linestyle='--', color='k', alpha=0.5)
    plt.xlabel('target')
    plt.ylabel('prediction')
    plt.plot((coord_start,coord_end),(coord_start,coord_end), '--', color='r', alpha=0.3)
    plt.xlim(coord_start,coord_end)
    plt.ylim(coord_start,coord_end)


def calc_r2_adj(r2, k, n):
    return 1 - (1-r2) * (k - 1)/(k - n - 1)


# ПОСТРОЕНИЕ ДЕНДРОГРАММЫ (компиляция нескольких функций с просторов интернета)
def plot_dendrogram(model, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    ddata = dendrogram(linkage_matrix, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')

# ПОСТРОЕНИЕ ГРАФИКА ИЗМЕНЕНИЯ РАССТОЯНИЙ МЕЖДУ КЛАСТЕРАМИ
def plot_linkage(model):
    counts = np.zeros(model.children_.shape[0])
    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    plt.figure()
    last = linkage_matrix[-40:, 2] # берем расстояние, последние значения в обратном порядке
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last_rev)

    acceleration = np.diff(last, 2)
    acceleration_rev = acceleration[::-1]
    plt.plot(idxs[:-2] + 1, acceleration_rev)

    plt.xlabel('cluster num')
    plt.ylabel('distance')
    plt.grid()

    k = acceleration_rev.argmax() + 2
    print('Recommended clusters num = ', k)
    
    
# ФУНКЦИЯ ДЛЯ КЛАСТЕРИЗАЦИИ
def AgglClustering(dataNorm, n_clust, plt_dendrogram = False):
    clusters = AgglomerativeClustering(distance_threshold=None, n_clusters=n_clust).fit(dataNorm)

    # к оригинальным данным добавляем номер кластера
    print(clusters.labels_)
    dataNorm['I'] = clusters.labels_
    
    # общая информация по кластерам
    res = dataNorm.groupby('I').mean()
    res['obj_num_AC'] = dataNorm.groupby('I').size().values
    print(res) 

    # статистики по кластерам
    for i in range(n_clust):
        print('cluster_num = ', i)
        print(dataNorm[dataNorm['I'] == i].describe())
        print('\n')

    if plt_dendrogram:
        plt.figure()
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(dataNorm)
        plot_dendrogram(model,
                        truncate_mode='lastp',
                        leaf_rotation=90.,
                        leaf_font_size=12.,
                        show_contracted=True,
                        annotate_above=10)

        plot_linkage(model)
    return clusters

# ФУНКЦИЯ ДЛЯ СОЗДАНИЯ ДАТАФРЕЙМОВ В ЗАВИСИМОСТИ ОТ КОЛ-ВА ИНТЕРЕСУЮЩИХ НАС ПРИЗНАКОВ СОГАЛСНО FEATURE IMPORTANCE
def df_generation_based_on_amount_of_top_features(amount, X_train, X_test, imp):
    
    #print(f"Кол-во фитчей до: {X_train.shape[1]}")
    #print(f"Кол-во топ фитчей: {amount}")
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    feature_names = pd.DataFrame(imp['feature_names']).rename(columns={0:'feature_names'})
    feature_importance = pd.DataFrame(imp['feature_importance']).rename(columns={0:'feature_importance'})

    df = pd.concat([feature_names, feature_importance], axis = 1).sort_values(by = 'feature_importance', ascending = False)
    df = df.head(amount)
    
    list_of_new_columns = [column for column in (df.head(amount).feature_names.to_list()) if (column in list(X_train.columns))]
    X_train_new = X_train[list_of_new_columns]
    X_test_new = X_test[list_of_new_columns]

    #print(f"Кол-во фитчей после: { X_train_new.shape[1]}")
    
    return X_train_new, X_test_new