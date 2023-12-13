from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import numpy as np
import re
from time import time

HIDDEN_PARAMS = [(100,100,100), (100,80,60), (150, 120, 100), (150,100)]
ALPHA = [1e-3, 1e-4, 1e-5]
ESTIMATORS = [50,100,150,200]
MAX_DEPTH =[3,5,7]
MIN_LEAF = [50,75,100]

def metrics(model, x, y):
    prediction = model.predict(x)

    f1 = f1_score(y, prediction, average='macro')
    precision = precision_score(y, prediction, average='macro')
    recall = recall_score(y, prediction, average='macro')
    accuracy = accuracy_score(y, prediction)

    scores = {'f1' : f1, 'precision' : precision, 'recall' : recall, 'accuracy' : accuracy}

    return scores

def run_MLP(trainx, trainy, testx, testy):
    hidden_layers = []
    activations = ['logistic', 'tanh', 'relu']
    alphas = []
    def tune_nn(x, y, hiddenparams, actparams, alphaparams):

        grid = {'hidden_layer_sizes': hiddenparams, 'activation': actparams, 'alpha': alphaparams}

        nn = MLPClassifier()

        grid_search = GridSearchCV(estimator=nn, param_grid=grid, scoring='accuracy')

        grid_search.fit(x, y)

        besthidden = grid_search.best_params_['hidden_layer_sizes']
        bestactivation = grid_search.best_params_['activation']
        bestalpha = grid_search.best_params_['alpha']

        results = {'best-hidden': besthidden, 'best-activation': bestactivation, 'best-alpha': bestalpha}
        return results

    best_results = tune_nn(trainx, trainy, HIDDEN_PARAMS, activations, ALPHA)
    hidden = best_results['best-hidden']
    activation = best_results['best-activation']
    alpha = best_results['best-alpha']
    nn = MLPClassifier(hidden_layer_sizes=hidden, activation=activation, alpha=alpha)

    nn.fit(trainx, trainy)

    # metrics for test
    test_metrics = metrics(nn, testx, testy)
    test_metrics['best-params'] = best_results
    return test_metrics

def run_RF(trainx, trainy, testx, testy):
    def tune_rf(x, y, num_estimators, dparams, lsparams):
        # 2a
        grid = {'n_estimators' : num_estimators, 'max_depth': dparams, 'min_samples_leaf': lsparams}
        rf = RandomForestClassifier()
        grid_search = GridSearchCV(estimator=rf, param_grid=grid, scoring='accuracy')
        grid_search.fit(x, y)
        best_estimator = grid_search.best_params_['n_estimators']
        best_depth = grid_search.best_params_['max_depth']
        best_leaf_samples = grid_search.best_params_['min_samples_leaf']
        results = {'best-estimator' : best_estimator, 'best-depth': best_depth, 'best-leaf-samples': best_leaf_samples}
        return results

    best_results = tune_rf(trainx, trainy, ESTIMATORS, MAX_DEPTH, MIN_LEAF)
    estimator = best_results['best-estimator']
    max_depth = best_results['best-depth']
    leaves = best_results['best-leaf-samples']
    forest = RandomForestClassifier(n_estimators=estimator, max_depth=max_depth, min_samples_leaf=leaves)

    forest.fit(trainx, trainy)
    # metrics for test
    test_metrics = metrics(forest, testx, testy)
    test_metrics['best-params'] = best_results
    return test_metrics

if __name__ == "__main__":
    input_csv = 'mega_data_3.csv'
    df = pd.read_csv(input_csv)
    
    y = df['genre']

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    
    # splitting data

    midi_train = X_train.iloc[:, 26:282]
    midi_test = X_test.iloc[:, 26:282]
    lyrics_train = X_train.iloc[:, 282:]
    lyrics_test = X_test.iloc[:, 282:]

    audio_indices = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    audio_metadata_train = X_train.iloc[:, audio_indices].copy()
    audio_metadata_test = X_test.iloc[:, audio_indices].copy()

    # for 3 mode
    trainx = audio_metadata_train.join(lyrics_train)
    trainx = trainx.join(midi_train)
    testx = audio_metadata_test.join(lyrics_test)
    testx = testx.join(midi_test)

    # lyrics_midi_trainx = lyrics_train.join(midi_train)
    # lyrics_midi_testx = lyrics_test.join(midi_test)
    #
    # lyrics_meta_trainx = lyrics_train.join(audio_metadata_train)
    # lyrics_meta_testx = lyrics_test.join(audio_metadata_test)
    #
    # meta_midi_trainx = audio_metadata_train.join(midi_train)
    # meta_midi_testx = audio_metadata_test.join(midi_test)

    audio_metadata_train = audio_metadata_train.values
    audio_metadata_test = audio_metadata_test.values

    # run models

    # 1 mode
    start = time()
    lyrics_results = run_MLP(lyrics_train, y_train, lyrics_test, y_test)
    duration = time() - start
    print("Lyrics complete. Time elapsed :" + str(duration))
    start = time()
    print("Results:")
    print(lyrics_results)
    metadata_results = run_MLP(audio_metadata_train, y_train, audio_metadata_test, y_test)
    duration = time() - start
    print("Metadata complete. Time elapsed :" + str(duration))
    print("Results:")
    print(metadata_results)
    start = time()
    midi_results = run_MLP(midi_train, y_train, midi_test, y_test)
    duration = time() - start
    print("MIDI complete. Time elapsed :" + str(duration))
    print("Results:")
    print(midi_results)

    # 2 mode
    # start = time()
    # lyrics_midi_results = run_MLP(lyrics_midi_trainx, y_train, lyrics_midi_testx, y_test)
    # duration = time() - start
    # print("Lyrics+MIDI complete. Time elapsed :" + str(duration))
    # start = time()
    # print("Results:")
    # print(lyrics_midi_results)
    #
    # lyrics_meta_results = run_MLP(lyrics_meta_trainx, y_train, lyrics_meta_testx, y_test)
    # duration = time() - start
    # print("Lyrics+metadata complete. Time elapsed :" + str(duration))
    # print("Results:")
    # print(lyrics_meta_results)
    #
    # start = time()
    # meta_midi_results = run_MLP(meta_midi_trainx, y_train, meta_midi_testx, y_test)
    # duration = time() - start
    # print("Metadata+MIDI complete. Time elapsed :" + str(duration))
    # print("Results:")
    # print(meta_midi_results)

    # 3 mode
    start = time()
    full_results = run_MLP(trainx, y_train, testx, y_test)
    duration = time() - start
    print("All three modes complete. Time elapsed :" + str(duration))
    print("Results:")
    print(full_results)


# RF results
# lyrics
#{'f1': 0.1638173266109198, 'precision': 0.557168458781362, 'recall': 0.172602202776976, 'accuracy': 0.47648686030428766, 'best-params': {'best-estimator': 50, 'best-depth': 7, 'best-leaf-samples': 50}}

# meta data
#{'f1': 0.14081647649084258, 'precision': 0.2476877953656796, 'recall': 0.1609768425501184, 'accuracy': 0.4598893499308437, 'best-params': {'best-estimator': 100, 'best-depth': 7, 'best-leaf-samples': 50}}

# midi
#{'f1': 0.40099179811723507, 'precision': 0.8059581917262393, 'recall': 0.3340212473571903, 'accuracy': 0.5947441217150761, 'best-params': {'best-estimator': 50, 'best-depth': 7, 'best-leaf-samples': 50}}

# lyrics + midi
# {'f1': 0.3253584904050201, 'precision': 0.6884447096260939, 'recall': 0.284712797951312, 'accuracy': 0.5719225449515906, 'best-params': {'best-estimator': 100, 'best-depth': 7, 'best-leaf-samples': 50}}

# lyrics + metadata
# {'f1': 0.16537813327564266, 'precision': 0.6466524289831206, 'recall': 0.17409665602694763, 'accuracy': 0.47925311203319504, 'best-params': {'best-estimator': 50, 'best-depth': 7, 'best-leaf-samples': 50}}

# metadata+midi
# {'f1': 0.3555321974319126, 'precision': 0.6839718286348326, 'recall': 0.30955426906536865, 'accuracy': 0.5899031811894883, 'best-params': {'best-estimator': 100, 'best-depth': 7, 'best-leaf-samples': 50}}

# all three
# {'f1': 0.3284857242009699, 'precision': 0.6776504665557314, 'recall': 0.28744426276119595, 'accuracy': 0.5691562932226832, 'best-params': {'best-estimator': 50, 'best-depth': 7, 'best-leaf-samples': 50}}


# NN results
# lyrics
# {'f1': 0.818694384174812, 'precision': 0.8178738723899861, 'recall': 0.826130772397797, 'accuracy': 0.843015214384509, 'best-params': {'best-hidden': (150, 100), 'best-activation': 'relu', 'best-alpha': 0.001}}

# meta data
# {'f1': 0.48714548408279035, 'precision': 0.5760512473441108, 'recall': 0.44638053707081427, 'accuracy': 0.5878284923928078, 'best-params': {'best-hidden': (150, 120, 100), 'best-activation': 'tanh', 'best-alpha': 1e-05}}

# midi
# {'f1': 0.733204234901485, 'precision': 0.7435944024005074, 'recall': 0.7388419973047803, 'accuracy': 0.7434301521438451, 'best-params': {'best-hidden': (150, 120, 100), 'best-activation': 'relu', 'best-alpha': 1e-05}}

# lyrics + midi
# {'f1': 0.6459075603164333, 'precision': 0.7076758575296176, 'recall': 0.6223381782365718, 'accuracy': 0.6604426002766252, 'best-params': {'best-hidden': (150, 120, 100), 'best-activation': 'relu', 'best-alpha': 0.0001}}

# lyrics + metadata
# {'f1': 0.7837470369402821, 'precision': 0.7857123803113849, 'recall': 0.7948230743463278, 'accuracy': 0.8035961272475796, 'best-params': {'best-hidden': (150, 120, 100), 'best-activation': 'tanh', 'best-alpha': 0.001}}

# metadata+midi
# {'f1': 0.6691078776652046, 'precision': 0.696806072020008, 'recall': 0.661258024073461, 'accuracy': 0.7040110650069157, 'best-params': {'best-hidden': (150, 120, 100), 'best-activation': 'relu', 'best-alpha': 1e-05}}

# all three
# {'f1': 0.7346408008427732, 'precision': 0.7340175038869149, 'recall': 0.7440045288998172, 'accuracy': 0.7461964038727524, 'best-params': {'best-hidden': (150, 120, 100), 'best-activation': 'relu', 'best-alpha': 0.001}}