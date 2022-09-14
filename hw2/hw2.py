import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 為了處理方便，把 'train.csv' 和 'test.csv' 合併起來，'test.csv'的 Weather 欄位用 0 補起來。
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_test['Label'] = np.zeros((len(df_test),))
# 以 train_end_idx 作為 'train.csv' 和 'test.csv' 分界列，
train_end_idx = len(df)
df = pd.concat([df, df_test], sort=False)

# label encode
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
df['Loc'] = labelEncoder.fit_transform(df['Loc'])
df['WindDir'] = labelEncoder.fit_transform(df['WindDir'])
df['DayWindDir'] = labelEncoder.fit_transform(df['DayWindDir'])
df['NightWindDir'] = labelEncoder.fit_transform(df['NightWindDir'])

# 將非數值欄位拿掉
df = df.drop(columns = [col for col in df.columns if df[col].dtype == np.object])

# KNN
# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=3)
# df_imputed = pd.DataFrame(imputer.fit_transform(df.drop(columns=['Label'])))

import sys
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
imputer = MissForest(max_iter=10)
df_imputed = pd.DataFrame(imputer.fit_transform(df.drop(columns=['Label'])))

# normalization
columns_label = df['Label'].reset_index(drop=True)
df_norm=(df_imputed - df_imputed.min()) / (df_imputed.max() - df_imputed.min())
df_done = pd.concat([df_norm, columns_label], axis=1)

# split data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    df_done.drop(columns = ['Label']).values[:train_end_idx, :],
    df_done['Label'].values[:train_end_idx], test_size=0.2)
X_test = df_done.drop(columns = ['Label']).values[train_end_idx:, :]

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
# 使用 chi2 score
# selection = SelectKBest(chi2, k=15).fit(X_train, y_train)

# deal with imbalance
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=0)
X_train, y_train = sm.fit_resample(X_train, y_train)

print('done')

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
#train tree model
model = ExtraTreesClassifier(n_estimators=400, max_depth=None, criterion='entropy', min_samples_split=10, min_samples_leaf=5, random_state=0)
model.fit(X_train,y_train)
#predict
y_pred_decision = model.predict(X_val)
print('Accuracy: %f' % accuracy_score(y_val, y_pred_decision))
print('f1-score: %f' % f1_score(y_val, y_pred_decision))
ans_pred = model.predict(X_test)
df_sap = pd.DataFrame(ans_pred.astype(int), columns = ['Label'])
df_sap.to_csv('myAns.csv',  index_label = 'Id')
