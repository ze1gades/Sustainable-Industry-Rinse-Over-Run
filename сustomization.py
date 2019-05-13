import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import time
plt.style.use('seaborn-darkgrid')
plt.rc('font', size=14)
warnings.filterwarnings("ignore")#отключение варнингов
pd.set_option('display.max_columns', None)


def metric(y_true, y_predict):
    N = len(y_predict)
    dif = np.abs(y_predict - y_true)
    dif /= np.maximum(np.abs(y_true), 290000)
    dif /= N
    MAPE = np.nansum(dif)
    return MAPE


train = pd.read_csv('type_3_train.csv', index_col=0)
train_labels = pd.read_csv('train_labels.csv',
                           index_col=0)
train_labels = train_labels.loc[train.index].values[:, 0]
print(train.shape, train_labels.shape)
X = train.values


start_time = time.monotonic()

pars = np.linspace(1000, 3000, 21).astype(int).tolist()
print(pars)
param_name = 'n_estimators'
model = GradientBoostingRegressor(random_state=0, loss='quantile', alpha=0.27, n_estimators=2000, max_depth=14,
                                  max_features=0.3, min_samples_split=0.04, min_samples_leaf=0.005, learning_rate=0.01)
from sklearn.model_selection import validation_curve
from sklearn.metrics import make_scorer
train_errors, test_errors = validation_curve(model, train, train_labels, param_name=param_name, param_range=pars,
                                             cv=3, scoring=make_scorer(metric), n_jobs=6)
plt.plot(pars, test_errors.mean(axis=1), label=str(time.monotonic() - start_time), lw=2)
plt.fill_between(pars, test_errors.min(axis=1), test_errors.max(axis=1), alpha=0.2, color="darkorange", lw=2)
plt.fill_between(pars, test_errors.mean(axis=1) - test_errors.std(axis=1),
                 test_errors.mean(axis=1) + test_errors.std(axis=1), alpha=0.1, color="darkblue", lw=1)
plt.legend(loc="best")
plt.ylabel('ROC AUC')
plt.xlabel(param_name)
plt.title(u'Качество при варьировании параметра')
plt.savefig("cross_val.png")
plt.show()

