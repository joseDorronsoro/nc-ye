import numpy as np

FULL_CLASS_LABELS = [0, 1, 2, 3, 4],

from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from data import (
    ye_targs, 
)

import fisher_functions as ff



def ye_classifier_targ(preds, pr):
    """Classifies predictions based on the closest target vector in the Ye embedding space, 
    given class probabilities pr.
    """
    l_class = []
    yt = ye_targs(pr)

    for i in range(preds.shape[0]):
        l_n = [np.linalg.norm(preds[i] - yt[ : , j]) for j in range(len(pr))]
        l_class.append(np.argmin(np.array(l_n)))

    return np.array(l_class)



def ohe_classifier(preds):
    """
    """
    return np.argmax(preds, axis=1)
    
    

def probs(n_classes, frac):
    """Returns imbalanced class probabilities assumin the first five are the 
    majority classes and the last five have a probability equal to frac times
    the majority probabilities.
    """
    class_prob = np.array(n_classes // 2 *[1.0] + n_classes // 2 * [frac])
    class_prob /= class_prob.sum()
        
    return class_prob

    
def majority_minority_accuracy(
        preds,
        targets,
        encoding,
        class_prob):
    """
    Compute separate accuracies for majority and minority classes.

    Targets and predictions are first converted to class labels
    using either OHE or Ye decoding.
    """
    if encoding == 'ye':
        pred_labels = ye_classifier_targ(preds, class_prob)
    
    else:
        pred_labels = np.argmax(preds, axis=1)
    
    targ_labels = np.argmax(targets, axis=1)

    majority = targ_labels <= 4
    minority = targ_labels > 4

    maj_acc = accuracy_score(
        targ_labels[majority],
        pred_labels[majority]
    )

    if minority.sum() > 0:
        min_acc = accuracy_score(
            targ_labels[minority],
            pred_labels[minority]
        )
    else:
        min_acc = 0.0

    return maj_acc, min_acc



def evaluate_model(
        model,
        train_loader,
        test_loader,
        encoding,
        device,
        bbn_predict):
    """
    Evaluate majority and minority class performance.

    Class probabilities are estimated from the training targets and
    used to decode Ye-encoded predictions when required.
    """
    _, train_targets = bbn_predict(
        model,
        train_loader,
        device
    )

    test_predictions, test_targets = bbn_predict(
        model,
        test_loader,
        device
    )

    _, counts = np.unique(
        train_targets,
        return_counts=True,
        axis=0
    )

    #assumes all majority and minority classes have same number of elements;
    #doesn't use probs above, as frac is not passed
    class_prob = np.sort(
        counts / counts.sum()
    )[::-1]
    #print('intermediate class_prob', class_prob)
    
    return majority_minority_accuracy(
        test_predictions,
        test_targets,
        encoding,
        class_prob,
    )


def evaluate_maj_min_preds(
        test_targets,
        test_predictions,
        frac,
        encoding,
        ):
        #ye_classifier):
    """
    Evaluate majority and minority class performance directly from model outputs.,
    i.e., without using model predicts.
    
    Class probabilities are computed and
    used to decode Ye-encoded predictions when required.
    """
    n_classes = test_targets.shape[1]
    
    if encoding == 'ye':
        class_prob = probs(n_classes, frac)
        #print('final', class_prob)
    elif encoding == 'ohe': 
        class_prob = None
    
    return majority_minority_accuracy(
        test_predictions,
        test_targets,
        encoding,
        class_prob,
    )


#review           
def nc_analysis(x_lhl, w_b_lhl, y, y_pred, verbose=False):
    """
    Compute and report Neural Collapse statistics on a dataset.

    Evaluates model performance together with NC1–NC4 metrics using
    hidden-layer representations, classifier weights, targets and
    model outputs.
    
    Ignores printing of NC2 and NC3 if verbose=False.
    Ignores probability correction in NC4.
    
    Criteria are those of Papyan's paper, that don't apply to imbalanced case,
    so left mostly as a reminder. 
    Can be ignored as a different analysis is needed in the imbalanced case.
    """
    print(10*'.' + ' model loss')
    print('mse:', mean_squared_error(y, y_pred))
    
    print(10*'.' + ' model acc')
    class_pred = np.argmax(np.array(y_pred), axis=1)
    
    if len(y.shape) == 2:
        num_classes = len(np.unique(y, axis=0))
        if y.shape[1] == num_classes:
            #Change targets to integer class labels
            y = np.argmax(y, axis=1)
        else:
            print("something is wrong ...")
    
    print("train conf matrix\n", confusion_matrix(y, class_pred))
    print("train_acc", accuracy_score(y, class_pred))
    print("balanced_acc", balanced_accuracy_score(y, class_pred))
            
    #NC1: class collpase to centroids, Sw near 0
    print(10*'.' + ' NCC1')
    ssw = ff.s_within(x_lhl, y) /x_lhl.shape[0]
    ssb = ff.s_between(x_lhl, y) /x_lhl.shape[0]
    pinv_ssb = np.linalg.pinv(ssb, rcond=1.e-4)
    print('max of Sw: values', np.abs(ssw).max())
    print('trace Sw_invSb', np.trace(ssw @ pinv_ssb))
    
    C = len(np.unique(y))
    ccm = ff.ccm_array(x_lhl, y)
        
    if verbose:
        #NC2: equinorm: centered class means, and lhl weights have the same norm
        print(10*'.' + ' NCC2')
        print('eq norm of centered class means', ff.equinorm(ccm))
        w_lhl = w_b_lhl[0]
        print('eq norm of lhl weights', ff.equinorm(w_lhl.T))
        
        #NC2: cosines of centered class means closs to simplex matrix
        C = len(np.unique(y))
        print('norm of ccm cos matrix - simplex matrix', np.linalg.norm(ff.cos_matrix(ccm) - ff.m_simplex(C)) / C / (C-1))
        
        #NC2: cosines of lhl weights close to simplex matrix
        print('norm of lhl weights cos matrix - simplex matrix', np.linalg.norm(ff.cos_matrix(w_lhl.T) - ff.m_simplex(C)) / C / (C-1))
        
        #NC3: approx self duality between centered means and lhl weights
        print(10*'.' + ' NCC3')
        norm_ccm = ccm / np.linalg.norm(ccm)
        norm_w_lhl = w_lhl.T / np.linalg.norm(w_lhl.T)
        
        #previous version
        #Papyan plots' version
        #print('duality of ccmeans and lhl weights', np.linalg.norm(norm_ccm - norm_w_lhl) / C / (C-1))
        
        print('duality of ccmeans and lhl weights', np.linalg.norm(norm_ccm - norm_w_lhl) **2.)
        
    #NCC4: equivalence of resnet preds with those of nearest class centers
    y_ncc_pred = np.zeros(x_lhl.shape[0])
    
    #nearest class center preds
    print(10*'.' + ' NCC4 (ignores probability correction)')
    
    for i in range(x_lhl.shape[0]):
        l_n = [np.linalg.norm(x_lhl[i] - ccm.T[j]) for j in range(C)]
        y_ncc_pred[i] = np.argmin(np.array(l_n))
    
    print("conf m of nearest class center prediction\n", 
           confusion_matrix(y, y_ncc_pred))
    print("acc of nearest class center prediction", 
           accuracy_score(y, y_ncc_pred))
    print("coincidence of model and ncc predictions", 
           accuracy_score(class_pred, y_ncc_pred))


def nc_analysis_test(
        x_lhl_tr,
        y_tr,
        x_lhl_ts,
        y_ts,
        y_pred_ts,
        encoding,
        frac):
    """
    Evaluate Neural Collapse statistics on test representations.

    Class centers are estimated from the training set and used to
    assess nearest-class-center behavior on test data.
    
    Ignores probability correction in NC4.
    
    Only confusion matrices and accuracies are relevant.
    """
    print(10*'.' + ' test model loss')
    print('mse:', mean_squared_error(y_ts, y_pred_ts))
    
    #get 1-dim test labels
    if len(y_tr.shape) == 2:
        num_classes = len(np.unique(y_tr, axis=0))
        if y_tr.shape[1] == num_classes:
            #Change targets to integer class labels
            y_tr = np.argmax(y_tr, axis=1)
            y_ts = np.argmax(y_ts, axis=1)
        else:
            print("something is wrong ...")
    
    print('\n' + 10 * '.' + ' test model acc')
    
    if encoding == 'ohe':
        class_pred = np.argmax(np.array(y_pred_ts), axis=1)
    elif encoding == 'ye':
        pr = probs(num_classes, frac)
        class_pred = ye_classifier_targ(y_pred_ts, pr)
    
    print("test_acc", accuracy_score(y_ts, class_pred))
    print("test balanced_acc", balanced_accuracy_score(y_ts, class_pred))
    
    print("test conf matrix\n", confusion_matrix(y_ts, class_pred))
            
    #test NCC4: equivalence of resnet preds with those of nearest class centers
    ccm = ff.ccm_array(x_lhl_tr, y_tr)
    
    #nearest class center preds
    y_ncc_pred = np.zeros(x_lhl_ts.shape[0])
    
    print(10*'.' + ' test NCC4 (ignores probability correction)')
        
    for i in range(x_lhl_ts.shape[0]):
        l_n = [np.linalg.norm(x_lhl_ts[i] - ccm.T[j]) for j in range(num_classes)]
        y_ncc_pred[i] = np.argmin(np.array(l_n))
    
    print("test conf m of nearest class center prediction\n", 
           confusion_matrix(y_ts, y_ncc_pred))
    print("test acc of nearest class center prediction", 
           accuracy_score(y_ts, y_ncc_pred))
    print("test coincidence of model and ncc predictions", 
           accuracy_score(class_pred, y_ncc_pred), flush=True)
                   

