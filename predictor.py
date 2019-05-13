import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm
from colorama import Fore
import time


train = [pd.read_csv('type_1_train.csv', index_col=0),
         pd.read_csv('type_2_train.csv', index_col=0),
         pd.read_csv('type_3_train.csv', index_col=0),
         pd.read_csv('type_6_train.csv', index_col=0),
         pd.read_csv('type_7_train.csv', index_col=0),
         pd.read_csv('type_8_train.csv', index_col=0),
         pd.read_csv('type_14_train.csv', index_col=0),
         pd.read_csv('type_15_train.csv', index_col=0)]

train_labels = pd.read_csv('train_labels.csv',
                           index_col=0)

submission_format = pd.read_csv('submission_format.csv', index_col=0)

test = [pd.read_csv('type_1_test.csv', index_col=0),
        pd.read_csv('type_2_test.csv', index_col=0),
        pd.read_csv('type_3_test.csv', index_col=0),
        pd.read_csv('type_6_test.csv', index_col=0),
        pd.read_csv('type_7_test.csv', index_col=0),
        pd.read_csv('type_8_test.csv', index_col=0),
        pd.read_csv('type_14_test.csv', index_col=0),
        pd.read_csv('type_15_test.csv', index_col=0)]

model = [GradientBoostingRegressor(random_state=0, loss='quantile', alpha=0.28, n_estimators=5000, max_depth=8,
                                   max_features=0.7, min_samples_split=0.06, min_samples_leaf=0.02,
                                   learning_rate=0.009),
         GradientBoostingRegressor(random_state=0, loss='quantile', alpha=0.3, n_estimators=1000, max_depth=8,
                                   max_features=0.3, min_samples_split=0.02, min_samples_leaf=0.006,
                                   learning_rate=0.03),
         GradientBoostingRegressor(random_state=0, loss='quantile', alpha=0.27, n_estimators=2000, max_depth=14,
                                   max_features=0.3, min_samples_split=0.04, min_samples_leaf=0.005,
                                   learning_rate=0.01),
         GradientBoostingRegressor(random_state=0, loss='quantile', alpha=0.27, max_depth=22, n_estimators=800,
                                   max_features=0.3, min_samples_split=0.048, min_samples_leaf=0.01,
                                   learning_rate=0.02),
         GradientBoostingRegressor(random_state=0, loss='quantile', alpha=0.25, n_estimators=5000, max_depth=10,
                                   max_features=0.3, min_samples_split=0.03, min_samples_leaf=0.02, learning_rate=0.03),
         GradientBoostingRegressor(random_state=0, loss='quantile', alpha=0.29, n_estimators=5000, max_depth=19,
                                   max_features=0.4, min_samples_split=0.05, min_samples_leaf=0.03,
                                   learning_rate=0.008),
         GradientBoostingRegressor(random_state=0, loss='quantile', alpha=0.29, n_estimators=10000, max_depth=13,
                                   max_features=0.1, min_samples_split=0.1, min_samples_leaf=0.03, learning_rate=0.009),
         GradientBoostingRegressor(random_state=0, loss='quantile', alpha=0.25, n_estimators=1000, max_depth=10,
                                   max_features=0.3, min_samples_split=0.02, min_samples_leaf=0.008, learning_rate=0.03)
         ]


preds = [1, 2, 3, 4, 5, 6, 7, 8]
for i in tqdm(range(8), bar_format="%s{l_bar}%s{bar}%s{r_bar}" % (Fore.RESET, Fore.LIGHTGREEN_EX, Fore.RESET)):
    model[i].fit(train[i], train_labels.loc[train[i].index].values[:, 0])
    pred = model[i].predict(test[i])
    preds[i] = pd.DataFrame(data=pred, columns=submission_format.columns, index=test[i].index)
pred = pd.concat(preds, axis=1)
pred.sort_index(inplace=True)

assert np.all(pred.index == submission_format.index)
my_submission = pd.DataFrame(data=pred.sum(axis=1), columns=submission_format.columns, index=submission_format.index)
print(my_submission.shape)
my_submission.to_csv('submission.csv')
