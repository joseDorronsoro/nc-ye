#!/usr/bin/env python
# coding: utf-8

import sys
import os
import joblib
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

def cos_matrix0(m):
    """compute cosines of the angles between the columns of m
    
    If NC holds, the std should be small; only menas are approx -1 /(C-1) for SGD training
    """
    m_cos = np.zeros((m.shape[1], m.shape[1]))
    for i in range(m.shape[1]):
        for j in range(m.shape[1]):
            with np.errstate(divide='raise'):
                try:
                    m_cos[i, j] = m.T[i].dot(m.T[j]) / np.linalg.norm(m.T[i]) / np.linalg.norm(m.T[j])
                #m_cos[i, j] = cosine_similarity(m.T[i].reshape(-1, 1), m.T[j].reshape(-1, 1))
                except FloatingPointError:
                    print('divisors', i, np.linalg.norm(m.T[i]), '\t', j, np.linalg.norm(m.T[j]))
                
    #print(np.abs(m_cos - cosine_similarity(m.T)).max())
    return cosine_similarity(m)


def cos_matrix(m):
    m = np.asarray(m, dtype=float)
    
    # Compute column norms
    norms = np.linalg.norm(m, axis=0)
    
    # Avoid division by zero
    norms[norms == 0] = 1.0
    
    # Normalize columns
    m_normalized = m / norms
    
    # Cosine similarity = dot product of normalized columns
    return m_normalized.T @ m_normalized
    

def class_means(X, y):
    """returns dict with class means
    """
    dim = X.shape[1]
    n_cl = len(np.unique(y))

    d_means = {} 
    for cl in range(n_cl):
        np_cl = (y == cl).sum()
        d_means[cl] = np.mean(X[y==cl], axis=0)
        
    return d_means


def equinorm(m):
    """check equinorm: compute the column norms of the matrix m
    and return std/mean
    """
    #norms of the diffs
    norm_diffs = np.linalg.norm(m, axis=0)
    #print(norm_diffs)
    #print('norm std, mean', norm_diffs.std() / norm_diffs.mean())
    return norm_diffs.std() / norm_diffs.mean()
    

def m_simplex(C):
    """compute simplex equiangular tight frame (ETF) matrix
    for C classes
    """
    m_simplex = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            m_simplex[i][j] = -1. / (C-1)
            if i == j:
                m_simplex[i][j]  += 1.
            
    return m_simplex

def s_between(X, y):
    """Unnormalized bewtween cov matrix
    """
    dim = X.shape[1]
    s_b = np.zeros((dim, dim))                  # will contain final scatter matrix
    #overall mean
    m = np.mean(X, axis=0).reshape(-1, 1) 
    n_cl = len(np.unique(y))

    #add s_b components
    for cl in range(n_cl):
        np_cl = (y == cl).sum()
        mv = np.mean(X[y==cl], axis=0).reshape(-1, 1)
        s_b += np_cl * (mv - m).dot((mv - m).T)
    
    return s_b


def s_total(x):
    """Unnormalized (ie, not divided by N) total cov matrix
    """
    #return x.shape[0] * np.cov(x, rowvar=False, bias=True)
    m = x.mean(axis=0)

    return (x-m).T @ (x-m)
    
    
def s_within(X, y):
    """Unnormalized within cov matrix. Requires y to be given as integer class labels
    """
    dim = X.shape[1]
    sw = np.zeros((dim, dim))                  # scatter matrix for every class
    
    labels, freqs = np.unique(y, return_counts=True)
    #print('llsh', len(labels.shape))
    #
    ##ensure y values are class labels; change it for fisher targs
    #if len(labels.shape) > 1:
    #    y = np.argmax(y, axis=1)
        
    for cl, fr in zip(range(len(labels)), freqs):
        ssv = fr * np.cov(X[y==cl], rowvar=False, bias=True)
        
        #second option: apply the direct formulae; coincides with cov based and not used
        #mm = X[y==cl].mean(axis=0)
        #ssv2 = (X[y==cl] - mm).T.dot(X[y==cl] - mm)
        #print(np.abs(ssv-ssv2).max())
        sw += ssv

    return sw


def centered_class_means0(X, y):
    """computes the differences between the overall and each class means
    and returns the ratio of their std to their mean.
    """
    #dict of class means
    m_cl = class_means(X, y)
    
    #overall mean
    m = np.mean(X, axis=0)
    
    # diffs between class and overall mean
    m_diffs = np.array([m_cl[k] - m.reshape(-1, ) for k in m_cl.keys()]).T

    return m_diffs


def centered_class_means(X, y):
    """Returns a dictionary of centered class means.

    For each class in `y`, computes the mean vector of samples in `X` belonging to that class,
    then subtracts the overall mean of `X` from each class mean. The result is a dictionary
    mapping class indices to their centered mean vectors.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Class labels, either as a 1D array or a 2D one-hot array.

    Returns:
        dict: Dictionary {class_index: centered_mean_vector} for each class.
    """
    #dict of class means
    d_clm = class_means(X, y)
    #print(d_clm[0].shape)
    
    #overall mean
    m = np.mean(X, axis=0)
    
    # diffs between class and overall mean
    for k in d_clm.keys():
    #m_diffs = np.array([m_cl[k] - m.reshape(-1, ) for k in m_cl.keys()])
        d_clm[k] = d_clm[k] - m.reshape(-1, )

    #print(d_clm.keys())
    #print(d_clm[0].shape)
    return d_clm


def frequencies(yy_labels, relative=0):
    """compute frequencies of the classes in the label rows
    and return the frequencies of the classes in these labels.
    """
    labels, freqs = np.unique(yy_labels, axis=0, return_counts=True)
    if relative:
        freqs = freqs / freqs.sum()
    return freqs


def h_pi(freqs, inverse=0):
    """
    Computes a diagonal matrix from the given frequencies.

    If `inverse` is set to a non-zero value, the function returns a diagonal matrix
    with the inverse of the frequencies.

    Args:
        freqs (array-like): A sequence of frequency values.
        inverse (int, optional): If non-zero, the function returns the inverse of the frequencies. Defaults to 0.

    Returns:
        numpy.ndarray: A diagonal matrix with the (inverse) frequencies.
    """
    if inverse:
        return np.diag(1. / freqs)
    else:
        return np.diag(freqs)
    

def freq_imbal(K, R):
    """compute frequency imbalance for K classes and R samples per class
    """
    p0 = R / (R + 1) / K
    p1 = 1 / (R + 1) / K

    freq_imbal = K * [p0] + K * [p1]
    
    return np.array(freq_imbal)
    

def etf_pi(freqs):
    """Compute the ETF-like matrix from the given frequencies."""
    if np.abs(1. - freqs.sum()) > np.finfo('float32').eps:
        print('no normalized freqs')
        return None
    
    return np.eye(len(freqs)) - np.sqrt(freqs).reshape(-1, 1) @ (np.sqrt(freqs).reshape(-1, 1)).T


def ccm_array0(x, y_labels, verbose=False):
    """Computes the centered class means (CCM) array for the given data and labels.
    This function calculates the centered mean vectors for each class in the dataset,
    based on the provided feature matrix and class labels. The result is returned as
    a 2D NumPy array where each column corresponds to the centered mean vector of a class.
    Args:
        x (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y_labels (array-like): Array of class labels of length n_samples.
        verbose (bool, optional): If True, prints additional information during computation. Defaults to False.
    Returns:
        np.ndarray: 2D array of shape (n_features, n_classes) containing centered class mean vectors.
    """
    d_cc = centered_class_means(x, y_labels)

    if verbose:
        print('ccm class labels:', sorted(d_cc.keys()))

    return np.array([d_cc[k] for k in sorted(list(d_cc.keys()))]).T


def ccm_array(x, y_labels, verbose=False):
#def ccm_array(x, y_labels):
    """Computes the centered class means (CCM) array for the given data and labels.
    This function calculates the centered mean vectors for each class in the dataset,
    based on the provided feature matrix and class labels. The result is returned as
    a 2D NumPy array where each column corresponds to the centered mean vector of a class.
    Args:
        x (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y_labels (array-like): Array of class labels of length n_samples.
        verbose (bool, optional): If True, prints additional information during computation. Defaults to False.
    Returns:
        np.ndarray: 2D array of shape (n_features, n_classes) containing centered class mean vectors.
    """
    #print('ccma', verbose)
    #d_cc = ff.class_means(x, y_labels, verbose=verbose)
    d_cc = centered_class_means(x, y_labels)

    if verbose:
        print('ccm class labels:', sorted(d_cc.keys()))

    return np.array([d_cc[k] for k in sorted(list(d_cc.keys()))]).T

