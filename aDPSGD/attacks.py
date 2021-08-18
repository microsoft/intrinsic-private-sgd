import statistics
from scipy.stats import bernoulli  # type: ignore
from sklearn.svm import LinearSVC

import ipdb

import numpy as np
# from utils.torch_utils import set_random_seeds, get_torch_device
# from utils.synthesis import create_data
# from model_interface.factory_methods import create_model_wrapper

# from feature_sets.independent_histograms import HistogramFeatureSet
# from feature_sets.model_agnostic import NaiveFeatureSet, EnsembleFeatureSet
# from feature_sets.bayes import CorrelationsFeatureSet

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


SPLIT_N = 5000


def get_threshold(train_loss):
    train_split = train_loss[SPLIT_N:]
    threshold = statistics.mean(train_split)

    return threshold


def get_classifier(train_act, test_act):
    """ Subset - use after SPLIT_N on both """
    train_split = train_act[SPLIT_N:]
    test_split = test_act[SPLIT_N:]
    print(f'Training classifier with {train_split.shape[0]} positive examples and {test_split.shape[0]} negative examples')
    d = train_split.shape[1]
    print(d)
    X = np.vstack([train_split, test_split])
    assert X.shape[1] == d
#    if d > 3000:
#        X = X[:, :500]
#        assert X.shape[1] == 500
    # train examples have label 1
    labels = np.array(([1] * train_split.shape[0]) + ([0] * test_split.shape[0]))
    assert labels.shape[0] == X.shape[0]

#    clf = Pipeline([('scaler', StandardScaler()),
#                    ('clf', SGDClassifier(loss='log',
#                                          class_weight='balanced')])
                    #                                          early_stopping=True))])
    #clf = Pipeline([('scaler', StandardScaler()),
    #                ('clf', MLPClassifier(hidden_layer_sizes=(64,),
    #                                      early_stopping=True))])
    clf = RandomForestClassifier(max_depth=5, class_weight='balanced')
#    clf = GradientBoostingClassifier()
    #clf = SGDClassifier(loss='log', class_weight='balanced', early_stopping=True)
    print('fitting classifier...')
    clf.fit(X, labels)
    print(f'train accuracy of fitted classifier: {clf.score(X, labels)}')
    return clf


def get_mi_attack_accuracy(train_loss, test_loss, classifier):
    """ split off the first 10k of both train and test """
    train_split = train_loss[0:SPLIT_N]
    test_split = test_loss[0:SPLIT_N]

    if type(classifier) in [float, np.float64, np.float32]:
        print('Interpreting classifier as fixed threshold...')
        threshold = classifier

        correct_mem = np.sum(train_split <= threshold)
        correct_nonmem = np.sum(test_split > threshold)
    else:
        print('Assuming classifier is classifier')
        correct_mem = np.sum(classifier.predict(train_split) == 1)
        correct_nonmem = np.sum(classifier.predict(test_split) == 0)
        # X = np.vstack([train_split, test_split])
        # true_labels = np.array(([1] * train_split.shape[0]) + ([0] * test_split.shape[0]))
        # d = X.shape[1]
        # small hack
#        if d > 3000:
#            X = X[:, :500]
#            assert X.shape[1] == 500
        # assert X.shape[0] == len(true_labels)
        # mi_attack_acc = classifier.score(X, true_labels)

    print("**Correct prediction for members", correct_mem, "out of", len(train_split))
    print("**Correct prediction for non-members", correct_nonmem, "out of", len(test_split))

    total = len(train_split) + len(test_split)
    correct_pred = correct_mem + correct_nonmem
    mi_attack_acc = correct_pred / total
    print(f'Attack accuracy: {mi_attack_acc}')
    return mi_attack_acc


def get_epsilon(mi_attack_acc):
    epsilon = np.log(mi_attack_acc/(1-mi_attack_acc))

    return epsilon

# #simple classifier based MI attack
# def simple_membership_inference(train_loss, test_loss, frac):#, clipping_norm, noise_multiplier, epochs, dataset_name, model_type):
#     '''
#     takes train_loss and test_loss (with labels 1 and 0 respectively) and fits a classifier to them
#     calculates: (a) classifier's overall accuracy, (b) the fraction of train and test samples in the dataset, (c) how accuracte the classifier is on predicting train and test samples individually
    
#     '''
  
#     train_split = train_loss[:len(test_loss)]
#     # th_max = max(train_split)
#     th_median = statistics.median(train_split)
#     th_mean = statistics.mean(train_split)


#     train_loss = train_loss[-len(test_loss):]

#     #print(len(train_loss), len(test_loss))

#     # correct_max = 0
#     correct_mean = 0
#     correct_median = 0
#     for i in range(0,len(train_loss)):
#         # if train_loss[i] < th_max:
#             # correct_max = correct_max + 1
#         if train_loss[i] < th_median:
#             correct_median = correct_median + 1       
#         if train_loss[i] < th_mean:
#             correct_mean = correct_mean + 1

#     for i in range(0,len(test_loss)):
#         # if test_loss[i] > th_max:
#             # correct_max = correct_max + 1
#         if test_loss[i] > th_median:
#             correct_median = correct_median + 1       
#         if test_loss[i] > th_mean:
#             correct_mean = correct_mean + 1

#     n = len(train_loss) + len(test_loss)

#     # Adv_max = correct_max / n
#     Adv_median = correct_median / n
#     Adv_mean = correct_mean / n

#     print('Membership attack accuracy with median and mean loss:',  Adv_median, Adv_mean)
#     #print('Membership attack accuracy with median',  Adv_median)

#     # eps_max = np.log(Adv_max/(1-Adv_max)) 
#     eps_median = np.log(Adv_median/(1-Adv_median)) 
#     eps_mean = np.log(Adv_mean/(1-Adv_mean)) 

#     ####  Test code to check the MI accuracy of the subset of train dataset 
#     correct_mean = 0
#     correct_median = 0

#     for i in range(0,len(train_split)):
#         # if train_loss[i] < th_max:
#             # correct_max = correct_max + 1
#         if train_split[i] < th_median:
#             correct_median = correct_median + 1       
#         if train_split[i] < th_mean:
#             correct_mean = correct_mean + 1

#     print("len of train_split and test loss", len(train_split), len(train_loss))

#     n = len(train_split) #+ len(test_loss)
#     Adv_median = correct_median / n
#     Adv_mean = correct_mean / n

#     print('MI accuracy of the original subset with median and mean loss:',  Adv_median, Adv_mean) #     return eps_median, eps_mean
