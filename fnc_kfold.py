import sys
import numpy as np
import nltk
nltk.download('wordnet')
from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from utils.system import parse_params, check_version
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
import textwrap
nltk.download('punkt')
from gensim.models.doc2vec import Doc2Vec
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
sys.setrecursionlimit(1500)
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import pickle
import timeit

def generate_vectors(b):
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in
                   enumerate(b)]
    model = init_doc2vec(tagged_data)
    X = model.docvecs.vectors_docs
    return X

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")
    X_body_vectors = generate_vectors(b)
    X_head_vectors = generate_vectors(h)
    #X_body_vectors = do_PCA(X_body_vectors, n_components=2)
    #X_head_vectors = do_PCA(X_head_vectors,n_components=2)
    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap, X_body_vectors, X_head_vectors]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    #X = do_PCA(X, n_components= 3)
    return X,y

def lstm_model(X_train, y_train):
    model = Sequential()
    model = Sequential()
    model.add(LSTM(256, input_shape=(X_train.shape[0], X_train.shape[1]), return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(LSTM(256))
    #model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.fit(X_train, y_train)
    return model

def ANN_classifier(X_train, y_train):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    model = Sequential()
    model.add(Dense(8, input_dim=44, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train,dummy_y, epochs=10, batch_size=64)
    return model

def generate_text_features(stances, dataset, name):
    h, b, y = [], [], []
    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])
    #X = np.vstack((h,b))
    X = np.c_[h, b]
    return X,y

def do_PCA(vectors, n_components = 2):
    pca = PCA(n_components = n_components)
    principalComponents = pca.fit_transform(vectors)
    return principalComponents

def init_doc2vec(tagged_data):
    max_epochs = 10
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    return model

if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    #print(d)
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    #X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition_test")
    '''
    X_comp_d = pd.DataFrame(X_competition)
    y_comp_d= pd.DataFrame(y_competition)
    X_comp_d.to_pickle('X_comp_d.pkl')
    y_comp_d.to_pickle('y_comp_d.pkl')
    '''
    #model = init_doc2vec(tagged_data)
    # save the model for later use
    #model.save("d2v.model")
    #print("Model Saved")
    #model = Doc2Vec.load("d2v.model")
    # model= Doc2Vec.load("d2v.model")
    # Get the vectors
    #X_competition = model.docvecs.vectors_docs
    #relation_vectors = do_PCA(textVect, n_components=2)
    #print(X_competition)
    #print(X_competition.shape)
    #print(y_competition)
    #print(y_competition.shape)
    #exit()
    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    '''
    X_holdout_d = pd.DataFrame(X_competition)
    y_holdout_d = pd.DataFrame(y_competition)
    X_holdout_d.to_pickle('X_holdout_d.pkl')
    y_holdout_d.to_pickle('y_holdout_d.pkl')
    '''
    # model = init_doc2vec(tagged_data)
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))
        #tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in
                       #enumerate(Xs[fold][:,1])]
        #model = init_doc2vec(tagged_data)
        #model.save("d2v1.model")
        #print("Model Saved")
        # model= Doc2Vec.load("d2v.model")
        # Get the vectors
        #Xs[fold] = model.docvecs.vectors_docs
    '''
    output = open('Xs.pkl', 'wb')
    pickle.dump(Xs, output)
    output = open('ys.pkl', 'wb')
    pickle.dump(ys, output)
    output.close()
    pkl_file = open('Xs.pkl', 'rb')
    Xs = pickle.load(pkl_file)
    pkl_file = open('ys.pkl', 'rb')
    ys = pickle.load(pkl_file)
    pkl_file.close()
    '''
    best_score = 0
    best_fold = None

    start = timeit.default_timer()
    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]
        #scaler = MinMaxScaler()
        #X_train = scaler.fit_transform(X_train)
        #X_test = scaler.fit_transform(X_test)
        #clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,5), random_state=14128)
        clf = ANN_classifier(X_train, y_train)
        #clf = lstm_model(X_train, y_train)

        #TODO
        #More classifiers
        #clf.fit(X_train, y_train)

        #predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        predicted = [LABELS[int(a)] for a in clf.predict_classes(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf

    end = timeit.default_timer()
    print('training time took : ', end-start)

    #Run on Holdout set and report the final score on the holdout set
    #X_holdout = scaler.fit_transform(X_holdout)
    predicted = [LABELS[int(a)] for a in clf.predict_classes(X_holdout)]
    #predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")

    #Run on competition dataset
    #X_competition = scaler.fit_transform(X_competition)
    predicted = [LABELS[int(a)] for a in clf.predict_classes(X_holdout)]
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual,predicted)
