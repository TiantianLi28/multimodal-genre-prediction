from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd


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
    input_csv = ''
    df = pd.read_csv(input_csv)
    # output
    y  = df['genre']
    # splitting data — todo get the subset of metadata and midi stuff
    lyrics = df['lyrics']

    # run models