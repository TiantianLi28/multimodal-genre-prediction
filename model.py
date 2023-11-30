from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import re

HIDDEN_PARAMS = [(100,100,100), (100,80,60), (150, 120, 100), (150,100)]
ALPHA = [1e-3, 1e-4, 1e-5]
ESTIMATORS = [50,100,150,200]
MAX_DEPTH =[3,5,7,10]
MIN_LEAF = [25,50,75,100]

def metrics(model, x, y):
    prediction = model.predict(x)

    f1 = f1_score(y, prediction)
    precision = precision_score(y, prediction)
    recall = recall_score(y, prediction)

    scores = {'f1' : f1, 'precision' : precision, 'recall' : recall}

    return scores

def run_MLP(trainx, trainy, testx, testy):
    hidden_layers = []
    activations = ['logistic', 'tanh', 'relu']
    alphas = []
    def tune_nn(x, y, hiddenparams, actparams, alphaparams):

        grid = {'hidden_layer_sizes': hiddenparams, 'activation': actparams, 'alpha': alphaparams}

        nn = MLPClassifier()

        grid_search = GridSearchCV(estimator=nn, param_grid=grid, scoring=['precision', 'recall'])

        grid_search.fit(x, y)

        besthidden = grid_search.best_params_['hidden_layer_sizes']
        bestactivation = grid_search.best_params_['activation']
        bestalpha = grid_search.best_params_['alpha']

        results = {'best-hidden': besthidden, 'best-activation': bestactivation, 'best-alpha': bestalpha}
        return results

    best_results = tune_nn(trainx, trainy, hidden_layers, activations, alphas)
    hidden = best_results['best-hidden']
    activation = best_results['best-activation']
    alpha = best_results['best-alpha']
    nn = MLPClassifier(hidden_layer_sizes=hidden, activation=activation, alpha=alpha)

    nn.fit(trainx, trainy)

    # metrics for test
    test_metrics = metrics(nn, testx, testy)
    return test_metrics

def run_RF(trainx, trainy, testx, testy):
    num_estimators = []
    max_depths = []
    min_samples_leaf = []
    def tune_rf(x, y, num_estimators, dparams, lsparams):
        # 2a
        grid = {'n_estimators' : num_estimators, 'max_depth': dparams, 'min_samples_leaf': lsparams}
        rf = RandomForestClassifier()
        grid_search = GridSearchCV(estimator=rf, param_grid=grid, scoring=['precision', 'recall'])
        grid_search.fit(x, y)
        best_estimator = grid_search.best_params_['n_estimators']
        best_depth = grid_search.best_params_['max_depth']
        best_leaf_samples = grid_search.best_params_['min_samples_leaf']
        results = {'best-estimator' : best_estimator, 'best-depth': best_depth, 'best-leaf-samples': best_leaf_samples}
        return results

    grid, best_results = tune_rf(trainx, trainy, num_estimators, max_depths, min_samples_leaf)
    estimator = best_results['best-estimator']
    max_depth = best_results['best-depth']
    leaves = best_results['best-leaf-samples']
    forest = RandomForestClassifier(n_estimators=estimator, max_depth=max_depth, min_samples_leaf=leaves)

    forest.fit(trainx, trainy)
    # metrics for test
    test_metrics = metrics(forest, testx, testy)
    return test_metrics

if __name__ == "__main__":
    input_csv = 'mega_data.csv'
    df = pd.read_csv(input_csv)
    
    y = df['genre']

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    
    # splitting data

    lyrics_train = X_train.iloc[:, 26:410]
    lyrics_test = X_test.iloc[:, 26:410]
    midi_train = X_train.iloc[:, 410:]
    midi_test = X_test.iloc[:, 410:]

    audio_indices = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    audio_metadata_train = X_train.iloc[:, audio_indices].copy()
    audio_metadata_test = X_test.iloc[:, audio_indices].copy()
    audio_metadata_train = audio_metadata_train.values
    audio_metadata_test = audio_metadata_test.values

    # run models