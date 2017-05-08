from create_data import create_dataset
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
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

def quality(X_test, y_test, estimator):
    # Test data accuracy score
    print "Test score:", estimator.score(X_test, y_test)

    # confusion matrix
    y_predict = rf.predict(X_test)
    print "Test Confusion matrix:\n"
    print confusion_matrix(y_test, y_predict)

    # Precision/Recall
    print "Test Precision:", precision_score(y_test, y_predict)
    print "Test Recall:", recall_score(y_test, y_predict)

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
    gd_grid_search = grid_search(GradientBoostingClassifier(), gd_grid, train_x, train_y,num_folds=cv_num)
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

def feature_importance(X_test, y_test, estimator, df):
    #Use sklearn's model to get the feature importances
    feature_importances = np.argsort(estimator.feature_importances_)
    print "top five:", list(df.columns[feature_importances[-1:-6:-1]])

    n = 10 # top 10 features

    importances = estimator.feature_importances_[:n]
    std = np.std([tree.feature_importances_ for tree in estimator.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(n):
        print("%d. %s (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title(estimator.__class__.__name__.+" Feature Importances")
    plt.bar(range(10), importances[indices], yerr=std[indices], color="r", align="center")
    plt.xticks(range(10), indices)
    plt.xlim([-1, 10])
    plt.show()

def main():
    df = create_dataset()
    X, y = feature_target_split(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    rf_best, gd_best, ada_best = run_models(X_train, y_train, cv_num=10)

    print "\n"+rf_best.__class__.__name__
    quality(X_test, y_test, rf_best)
    feature_importance(X_test, y_test, gd_best, df)

    print "\n"+gd_best.__class__.__name__
    quality(X_test, y_test, gd_best)
    feature_importance(X_test, y_test, rf_best, df)

    print "\n"+ada_best.__class__.__name__
    quality(X_test, y_test, ada_best)
    feature_importance(X_test, y_test, ada_best, df)

if __name__ == '__main__':
    df = create_dataset()
    X, y = feature_target_split(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    rf_best, gd_best, ada_best = run_models(X_train, y_train, cv_num=10)

    print "\n"+rf_best.__class__.__name__
    quality(X_test, y_test, rf_best)
    feature_importance(X_test, y_test, gd_best, df)

    print "\n"+gd_best.__class__.__name__
    quality(X_test, y_test, gd_best)
    feature_importance(X_test, y_test, rf_best, df)

    print "\n"+ada_best.__class__.__name__
    quality(X_test, y_test, ada_best)
    feature_importance(X_test, y_test, ada_best, df)
