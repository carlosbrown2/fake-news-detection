#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:19:16 2019

@author: carlosbrown

Useful data science functions and classes
"""

"""STATISTICS"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import scipy.stats as stats
from nltk.tokenize import RegexpTokenizer
from collections import Counter

def draw_perm_reps(data_1, data_2, func, size=1):
        """Generate multiple permutation replicates."""

        # Initialize array of replicates: perm_replicates
        perm_replicates = np.empty(size)

        for i in range(size):
            # Generate permutation sample
            perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

            # Compute the test statistic
            perm_replicates[i] = func(perm_sample_1,perm_sample_2)

        return perm_replicates

def diff_of_means(sample1,sample2):
        return np.mean(sample1)-np.mean(sample2)

def permutation_sample(data1, data2):
        """Generate a permutation sample from two data sets."""

        # Concatenate the data sets: data
        data = np.concatenate((data1,data2))

        # Permute the concatenated array: permuted_data
        permuted_data = np.random.permutation(data)

        # Split the permuted array into two: perm_sample_1, perm_sample_2
        perm_sample_1 = permuted_data[:len(data1)]
        perm_sample_2 = permuted_data[len(data1):]

        return perm_sample_1, perm_sample_2


def draw_bs_reps(data, func, size=1):
        """Draw bootstrap replicates."""

        # Initialize array of replicates: bs_replicates
        bs_replicates = np.empty(shape=size)

        # Generate replicates
        for i in range(size):
            bs_replicates[i] = bootstrap_replicate_1d(data,func)

        return bs_replicates

def bootstrap_replicate_1d(data, func):
    """Draw bootstrap replicate"""
    
    boot_sample = np.random.choice(data,size=len(data),replace=True)
    
    return func(boot_sample)

def conf_int(data, conf=95):
    """Create bootstrap confidence interval of mean"""
    
    x1 = (100-conf) / 2
    x2 = 100 - x1
    boot_samples = draw_bs_reps(data, np.mean, size=10000)
    
    return np.percentile(boot_samples, [x1,x2])

def pearsonr_ci(x,y,alpha=0.05):
    """calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    """

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

"""MACHINE LEARNING"""

def evalclusters(data,ks=range(1,6)):
    """function to fit multiple KMeans models for evaluating inertia on one dataset"""
    inertias = []
    for k in ks:
        model = KMeans(n_clusters=k)
        model.fit(data)
        inertias.append(model.inertia_)
    # Plot ks vs inertias
    plt.plot(ks, inertias, '-o')
    lt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.title('Elbow Plot - KMeans Clustering')
    plt.xticks(ks)
    plt.show()

def cv_optimize(clf, parameters, Xtrain, ytrain, n_folds=5):
    """tune hyperparameters of untrained machine learning model using k-fold 
    cross validation"""
    gs = sklearn.model_selection.GridSearchCV(clf, param_grid=parameters, cv=n_folds)
    gs.fit(Xtrain, ytrain)
    print("BEST PARAMS", gs.best_params_)
    best = gs.best_estimator_
    return best


def do_classify(clf, parameters, indf, featurenames, targetname, target1val, standardize=False, train_size=0.8):
    """run classification"""
    """local dependencies: cv_optimize"""
    subdf=indf[featurenames]
    if standardize:
        subdfstd=(subdf - subdf.mean())/subdf.std()
    else:
        subdfstd=subdf
    X=subdfstd.values
    y=(indf[targetname].values==target1val)*1
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=train_size)
    clf = cv_optimize(clf, parameters, Xtrain, ytrain)
    clf=clf.fit(Xtrain, ytrain)
    training_accuracy = clf.score(Xtrain, ytrain)
    test_accuracy = clf.score(Xtest, ytest)
    print("Accuracy on training data: {:0.2f}".format(training_accuracy))
    print("Accuracy on test data:     {:0.2f}".format(test_accuracy))
    return clf, Xtrain, ytrain, Xtest, ytest

def getrocdata(clf,X_test,y_test):
    """calculate ROC curve and AUC of classifier"""
    
    y_pred_proba = clf.predict_proba(X_test)
    fpr, tpr, thresh = roc_curve(y_test, y_pred_proba[:,1])
    auc_ret = auc(fpr,tpr)
    print('Area under curve:',auc)
    return fpr, tpr, auc_ret

def plot_roc_curve(fpr, tpr): 
    """plots roc curve given false positive rate (fpr), and true positive rate (tpr)"""
    plt.plot(fpr, tpr, label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred,labels=classes)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    ax.grid('off')
    return ax

def cleanwordcounts(text):
    #returns words counts minus stopwords from nltk and chinese characters
    #expects pandas series or single string
    #CountVectorizer expects iterable over raw text documents, so transform text
    #if simple string
    if type(text)==str:
        text = [text,'']
    
    stpwords = stopwords.words('english')
    #instantiate countvectorizer
    tokens = CountVectorizer(stop_words=stpwords,analyzer='word',token_pattern=r'\w+')
    #transform corpus into vector space
    textvec = pd.DataFrame(tokens.fit_transform(text).toarray())
    #create counts dataframe
    counts = pd.DataFrame(list(zip(tokens.get_feature_names(),textvec.sum().values)),columns=['word','counts'])
    #remove any chinese characters
    counts['word'] = counts['word'].str.replace(r'[^\x00-\x7F]+', '')
    #drop rows where chinese characters residedp
    counts.drop(counts[counts['word']==''].index, inplace=True)
    return counts.sort_values(by=['counts'],ascending=False).reset_index(drop=True)

def wordcounts(text):
    #tokenize to words and return counts of each word, including stopwords
    if type(text) == pd.core.series.Series:
        text = text.str.cat(sep=' ')
    elif type(text) != str:
        print('Wrong data type passed to function')
        return 0
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    #lowercase
    tokens = [t.lower() for t in tokens]
    #create dict from Counter object
    dict_tok = dict(Counter(tokens))
    #convert to dataframe and return
    tok = pd.DataFrame.from_dict(data=dict_tok,orient='index').reset_index()
    tok.columns=['word','counts']
    return tok.sort_values(by='counts',ascending=False)
