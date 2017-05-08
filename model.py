from create_data import create_dataset
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV

def feature_target_split(df):
    y = df.pop('label')
    df.pop('ARR_DELAY_NEW')
    x = df.values
    return x, y

def grid_search(est, grid, train_x, train_y, num_folds=10):
    grid_cv = GridSearchCV(est, grid, n_jobs=-1, verbose=True,
                           scoring='f1', cv=num_folds).fit(train_x, train_y)
    return grid_cv

def cross_val(estimator, train_x, train_y, num_folds=10):
    # n_jobs=-1 uses all the cores on your machine
    f1 = cross_val_score(estimator, train_x, train_y,
                           scoring='f1',
                           cv=num_folds, n_jobs=-1)
    accuracy = cross_val_score(estimator, train_x, train_y,
                           scoring='accuracy', cv=num_folds, n_jobs=-1)

    precision = cross_val_score(estimator, train_x, train_y,
                           scoring='precision',
                           cv=num_folds, n_jobs=-1)
    recall = cross_val_score(estimator, train_x, train_y,
                           scoring='recall', cv=num_folds, n_jobs=-1)

    mean_f1 = f1.mean()
    mean_accuracy = accuracy.mean()
    mean_precision = precision.mean()
    mean_recall = recall.mean()

    params = estimator.get_params()
    name = estimator.__class__.__name__
    print '%s Train CV | f1: %.3f | accuracy: %.3f | precision: %.3f | recall: %.3f' % (name, mean_f1, mean_accuracy, mean_preicision, mean_recall)
    return mean_f1, mean_accuracy, mean_precision, mean_recall

def run_models(train_x, train_y, cv_num=10):
    rf_grid = {'max_depth': [3, 10,20],
           'max_features': [1, 3, 10],
           'min_samples_split': [1, 3, 10],
           'min_samples_leaf': [1, 3, 10],
           'bootstrap': [True, False],
           'n_estimators': [25, 40, 50],
           'random_state': [1]}

    gd_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
               'max_depth': [4, 6],
               'min_samples_leaf': [3, 5, 9, 17],
               'max_features': [1.0, 0.3, 0.1],
               'n_estimators': [500],
               'random_state': [1]}

    ada_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
               'max_depth': [4, 6],
               'min_samples_leaf': [3, 5, 9, 17],
               'max_features': [1.0, 0.3, 0.1],
               'n_estimators': [500],
               'random_state': [1]}

    rf_grid_search = grid_search(RandomForestRegressor(), rf_grid, train_x, train_y, num_folds=cv_num)
    gd_grid_search = grid_search(GradientBoostClassifier(), gd_grid, train_x, train_y,num_folds=cv_num)
    ada_grid_search = grid_search(AdaBoostClassifier, ada_grid, train_x, train_y,num_folds=cv_num)

    #Best models
    rf_best = rf_grid_search.best_estimator_
    gd_best = gd_grid_search.best_estimator_
    ada_best = ada_grid_search.best_estimator_

    cross_val(rf_best, train_x, train_y, num_folds=cv_num)
    cross_val(gd_best, train_x, train_y, num_folds=cv_num)
    cross_val(ada_best, train_x, train_y, num_folds=cv_num)

    return rf_best, gd_best, ada_best


def plot_feature_importances(model):
    colnames = get_spam_names()

    feat_import = model.feature_importances_

    top10_nx = np.argsort(feat_import)[::-1][0:10]

    feat_import = feat_import[top10_nx]
    #normalize:
    feat_import = feat_import /feat_import.max()
    colnames = colnames[top10_nx]

    x_ind = np.arange(10)

    plt.barh(x_ind, feat_import, height=.3, align='center')
    plt.ylim(x_ind.min() + .5, x_ind.max() + .5)
    plt.yticks(x_ind, colnames[x_ind])
    return top10_nx

def main():
    df = create_dataset()
    X, y = feature_target_split(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    rf_best, gd_best, ada_best = run_models(X_train, y_train, cv_num=10):

# def grid_search
# def kfold_model(model,num_folds = 5, x,y):
#     kfold = KFold(len(y), n_folds = num_folds)
#
#     accuracies = []
#     precisions = []
#     recalls = []
#
#     for train_index, test_index in kfold:
#         model.fit(x[train_index], y[train_index])
#         y_predict = model.predict(x[test_index])
#         y_true = y[test_index]
#         accuracies.append(accuracy_score(y_true, y_predict))
#         precisions.append(precision_score(y_true, y_predict))
#         recalls.append(recall_score(y_true, y_predict))
#
#     print "accuracye:", np.average(accuracies)
#     print "precision:", np.average(precisions)
#     print "recall:", np.average(recalls)


if __name__ == '__main__':
    df = create_dataset()
    x, y = feature_target_split(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    rf = RandomForestClassifier()
    kfold_rf(model,num_folds = 5, X_train,y_train)
