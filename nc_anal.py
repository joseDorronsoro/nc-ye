#!/usr/bin/env python
# coding: utf-8
########################################################################################################
import sys
import argparse
import numpy as np
import joblib

from sklearn.metrics import accuracy_score

#import fisher_functions as ff

# -------- CLI Arguments & Parameters ------------------------------------------
parser = argparse.ArgumentParser(
    description="Analyze Neural Collapse / UFM features and subspace alignments."
)

parser.add_argument("--fr", type=float, default=0.01, help="Imbalance fraction (default: 0.01)")
parser.add_argument("--opt", type=str, default="sgd", choices=["sgd", "adamw"], help="Optimizer choice (default: sgd)")
parser.add_argument("--rnm", type=str, default="18", help="Resnet model; (default: 18)")
parser.add_argument("--lrf", type=float, default=None, help="Learning rate factor (default: 1.0 for sgd, 0.01 for adamw)")
parser.add_argument("--bs", type=int, default=192, help="Batch size (default: 192)")
parser.add_argument("--wd", type=float, default=0.0005, help="Weight decay (default: 0.0005)")
parser.add_argument("--hrd", type=float, default=0., help="Feature norm regularization lambda_h (default: 0.)")
parser.add_argument("--epochs", type=int, default=350, help="Total epochs (default: 600)")
parser.add_argument("--warm_epochs", type=int, default=20, help="Warmup epochs (default: 20)")
parser.add_argument("--l_anc", type=float, default=0., help="Anchor loss weight lambda_anc (default: 0.)")
parser.add_argument("--l_center", type=float, default=0., help="Center loss weight lambda_center (default: 0.)")
parser.add_argument("--results_dir", type=str, default="../nc-ye_exps/", help="Directory containing joblib files (default: ../nc-ye_exps/)")

args = parser.parse_args()

# Extract parameters
fr = args.fr
rnm = args.rnm
opt = args.opt
bs = args.bs
wd = args.wd
hrd = args.hrd
epochs = args.epochs
warm_epochs = args.warm_epochs
l_anc = args.l_anc
l_center = args.l_center
results_dir = args.results_dir

# Handle conditional defaults for learning rate factor
if args.lrf is not None:
    lrf = args.lrf
else:
    lrf = 1.0 if opt == "sgd" else 0.01

# ------- frozen weight matrix ------------------------------------------------
def probs(frac):
    """Imbalanced probabilities"""
    pr = np.array(5 * [1.] + 5 * [float(frac)])
    pr = pr / pr.sum()
    return pr 


def householder_V(pr):
    """Householder V matrix for the SVD of h_pi"""
    v = (np.sqrt(pr) + np.array(9 * [0.] + [1.])).reshape(-1, 1)
    V = (2. / (v.T @ v)) * (v @ v.T) - np.eye(len(pr))
    return V


def h_pi(pr):
    """H_pi matrix from probs."""
    H_pi = np.eye(10) - np.sqrt(pr).reshape(-1, 1) @ np.sqrt(pr).reshape(-1, 1).T
    return H_pi


def optimal_W(dim, frac, verbose=False):
    """Frozen weight matrix for Y targets"""
    pr = probs(frac)
    V = householder_V(pr)
    
    J = np.eye(len(pr))
    J[-1, -1] = 0.
    
    U = np.eye(dim)[:, : len(pr)]
    
    if verbose:
        print('\n' + 10 * '.' + ' checking optimal weight matrix')
        print('U.T @ U:', np.allclose(U.T @ U, np.eye(len(pr))))
        print('V @ V.T:', np.allclose(V @ V.T, np.eye(len(pr))))
        print('diag J', np.diagonal(J))
        print('h_pi svd', np.allclose(V @ J @ V.T, h_pi(pr)), '\n')
    
    return U @ J @ V.T

# -----------------------------------------------------------------------------
# subspace projection function
def project_Q_2_W(Q, W):
    Q_proj = W @ np.linalg.pinv(W, rcond=1.e-3) @ Q
    return Q_proj
    
    
def same_subspace_residual(A: np.ndarray, B: np.ndarray, tol: float = 1e-5) -> bool:
    """Checks if col(A) == col(B) by projecting B onto the column space of A."""
    UA, SA, _ = np.linalg.svd(A, full_matrices=False)
    UB, SB, _ = np.linalg.svd(B, full_matrices=False)
    
    rank_A = np.sum(SA > tol)
    rank_B = np.sum(SB > tol)
    
    if rank_A != rank_B:
        return False
    
    UA_k = UA[:, :rank_A]
    
    proj_B = UA_k @ (UA_k.T @ B)
    residual = np.linalg.norm(B - proj_B, ord='fro')
    
    return float(residual) < tol
    
    
def subspace_min_cosine(A: np.ndarray, B: np.ndarray, tol: float = 1e-5) -> float:
    """Returns the cosine of the worst-case principal angle between col(A) and col(B)."""
    UA, SA, _ = np.linalg.svd(A, full_matrices=False)
    UB, SB, _ = np.linalg.svd(B, full_matrices=False)
    
    rank_A = np.sum(SA > tol)
    rank_B = np.sum(SB > tol)
     
    common_rank = min(rank_A, rank_B)
    UA_k = UA[:, :common_rank]
    UB_k = UB[:, :common_rank]
    
    cos_thetas = np.linalg.svd(UA_k.T @ UB_k, compute_uv=False)
    
    return cos_thetas
    #return float(np.min(cos_thetas))
    

def orthogonal_procrustes(A, B):
    """Returns an orthonornal matrix R auch that ||A - R @ B|| is miminal.
    R @ B rotates B so that it is closest to A.
    """
    M = A @ B.T
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    return R
    
    
def cosines_norms(a):
    aa = a.T @ a
    na = np.linalg.norm(a, axis=0)
    return aa / (na.reshape(-1, 1) @ na.reshape(-1, 1).T), na


def cos_norms_theo(pr):
    cos_column = (np.sqrt(pr) / np.sqrt(1. - pr)).reshape(-1, 1)
    theo_cos_matrix = -cos_column @ cos_column.T
    theo_cos_matrix += np.eye(n_classes) + np.diag(pr / (1 - pr))
    theo_w_norms = np.sqrt(1. - pr)

    return theo_cos_matrix, theo_w_norms
        

# -----------------------------------------------------------------------------
# from fisher_functions
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
    
    #overall mean
    m = np.mean(X, axis=0)
    
    # diffs between class and overall mean
    for k in d_clm.keys():
        d_clm[k] = d_clm[k] - m.reshape(-1, )

    return d_clm


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


def ccm_array(x, y_labels, verbose=False):
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
    m = x.mean(axis=0)

    return (x-m).T @ (x-m)
    
    
def s_within(X, y):
    """Unnormalized within cov matrix. Requires y to be given as integer class labels
    """
    dim = X.shape[1]
    sw = np.zeros((dim, dim))                  # scatter matrix for every class
    
    labels, freqs = np.unique(y, return_counts=True)
        
    for cl, fr in zip(range(len(labels)), freqs):
        ssv = fr * np.cov(X[y==cl], rowvar=False, bias=True)
        sw += ssv

    return sw


# --------------------------------------------------------------- accuracies
def ye_targs(pr):
    """
    Construct Ye target vectors from class probabilities.
    """
    return (1. / np.sqrt(pr)) * np.eye(len(pr)) -  np.sqrt(pr).reshape(-1, 1) * np.ones(len(pr)).reshape(1, -1)


def ye_classifier_targ(preds, pr):
    """Classifies predictions based on the closest target vector in the Ye embedding space, 
    given class probabilities pr.
    """
    l_class = []
    yt = ye_targs(pr)

    for i in range(preds.shape[0]):
        l_n = [np.linalg.norm(preds[i] - yt[ : , j]) for j in range(len(pr))]
        l_class.append(np.argmin(np.array(l_n)))
        #l_class.append(np.argmax(preds[i]))

    return np.array(l_class)


def majority_minority_accuracy(
        preds,
        targets,
        class_prob,
        encoding='ye'):
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


# -----------------------------------------------------------------------------
# print params
print('\n' + 10 * '.' + 'params')
print(
    'opt = ', opt,
    '\tmodel = ', rnm,
    '\tfr = ', fr,
    '\tlrf = ', lrf,
    '\tbs = ', bs,
    '\twd = ', wd, 
    '\thrd = ', hrd,
    '\tepochs = ', epochs, 
    '\twarm_epochs = ', warm_epochs, 
    '\tl_anc = ', l_anc,
    '\tl_center = ', l_center
)

print(rnm)

# results path construction
# bunch_mnist_efm_ye_0.01_18_opbslrwd_sgd_192_1.0_0.0005_epwu_350_20_hrlalc_0.0_0.0_1e-06_0_train_results
file_str = (
    f"bunch_mnist_efm_ye_{fr}_{rnm}_opbslrwd_{opt}_{bs}_{lrf}_{wd}"
    f"_epwu_{epochs}_{warm_epochs}_hrlalc_{hrd}_{l_anc}_{l_center}_0"
)

#train outputs
file_path = f"{results_dir.rstrip('/')}/{file_str}_train_results.joblib"
print(f"\nLoading train results from: {file_path}")

targs_out, model_out, ll, w_b_lhl = joblib.load(file_path)

w, b = w_b_lhl
w, b = w.T, b.T

w_th = optimal_W(w.shape[0], fr)

label = np.argmax(targs_out, axis=1)
ccm = ccm_array(ll, label)

pr = probs(fr)
n_classes = len(pr)

q = ccm @ np.diag(np.sqrt(pr))
qp = np.linalg.pinv(q, rcond=1.e-5)
H_pi = h_pi(pr)

# NC1
print('\n' + 10 * '.' + ' traces')
s_with = s_within(ll, label) / ll.shape[0]
s_betw = s_between(ll, label) / ll.shape[0]
nc1_tr = np.trace(np.linalg.pinv(s_betw, rcond=1.e-5) @ s_with) / n_classes
print(f'trace (s_B^+ s_W) / C: {nc1_tr: .6f}')

# NC2 norms
## projections
pq = project_Q_2_W(q, w)
pw = project_Q_2_W(w, q)

## NC2 norms
print('\n' + 10 * '.' + ' NC2 norms')

cos_w, n_w = cosines_norms(w)
cos_pw, n_pw = cosines_norms(pw)
cos_q, n_q = cosines_norms(q)
cos_pq, n_pq = cosines_norms(pq)
cos_theo, n_theo = cos_norms_theo(pr)

print(f'    ||w|| - ||w||_teo: max {np.abs(n_w - n_theo).max(): 8.4f} \tmean {np.abs(n_w - n_theo).mean(): 8.4f}')
print(f'    ||q|| - ||w||_teo: max {np.abs(n_q - n_theo).max(): 8.4f} \tmean {np.abs(n_q - n_theo).mean(): 8.4f}')
print(f'  ||pq|| - ||w||_teo: max {np.abs(n_pq - n_theo).max(): 8.4f} \tmean {np.abs(n_pq - n_theo).mean(): 8.4f}')
print(f'  ||pw|| - ||w||_teo: max {np.abs(n_pw - n_theo).max(): 8.4f} \tmean {np.abs(n_pw - n_theo).mean(): 8.4f}')

# NC2 cosines
print('\n' + 10 * '.' + ' NC2 cosines')
print(f' |cos_w - cos_theo|: max {np.abs(cos_w - cos_theo).max(): 8.4f} \tmean {np.abs(cos_w - cos_theo).mean(): 8.4f}')
print(f' |cos_q - cos_theo|: max {np.abs(cos_q - cos_theo).max(): 8.4f} \tmean {np.abs(cos_q - cos_theo).mean(): 8.4f}')
print(f'|cos_pq - cos_theo|: max {np.abs(cos_pq - cos_theo).max(): 8.4f} \tmean {np.abs(cos_pq - cos_theo).mean(): 8.4f}')
print(f'|cos_pw - cos_theo|: max {np.abs(cos_pw - cos_theo).max(): 8.4f} \tmean {np.abs(cos_pw - cos_theo).mean(): 8.4f}')

# NC3: w, q collinearity  
print('\n' + 10 * '.' + ' NC3 w, pq collinearity')
w_q_norm_diff = np.linalg.norm(w /n_w - q / n_q, axis=0)
print(f'||norm_w|| - ||norm_q||: max {w_q_norm_diff.max(): 8.4f} \tmean {w_q_norm_diff.mean(): 8.4f}')
w_pq_norm_diff = np.linalg.norm(w /n_w - pq / n_pq, axis=0)
print(f'||norm_w|| - ||norm_pq||: max {w_pq_norm_diff.max(): 8.4f} \tmean {w_pq_norm_diff.mean(): 8.4f}')
pw_q_norm_diff = np.linalg.norm(pw /n_pw - q / n_q, axis=0)
print(f'||norm_pw|| - ||norm_q||: max {pw_q_norm_diff.max(): 8.4f} \tmean {pw_q_norm_diff.mean(): 8.4f}')

# projections and subspaces  
print('\n' + '\n' + 10 * '.' + 'Projected w, q vs H_pi')
print(f'w.T @ q - H_pi: \tmax abs diff {np.abs(w.T @ q - H_pi).max().round(3)} \tmean abs diff {np.abs(w.T @ q - H_pi).mean().round(3)}')
print(f'w_th.T @ w - H_pi: \tmax abs diff {np.abs(w_th.T @ w - H_pi).max().round(3)} \tmean abs diff {np.abs(w_th.T @ w - H_pi).mean().round(3)}')
print(f'w_th.T @ q - H_pi: \tmax abs diff {np.abs(w_th.T @ q - H_pi).max().round(3)} \tmean abs diff {np.abs(w_th.T @ q - H_pi).mean().round(3)}')
#print(f'\nw.T @ pq - H_pi: \tmax abs diff {np.abs(w.T @ pq - H_pi).max().round(3)} \tmean abs diff {np.abs(w.T @ pq - H_pi).mean().round(3)}')

print(f'\n||w.T @ q - H_pi||**2.: {np.linalg.norm(w.T @ q - H_pi) ** 2.: .4f}',  
      f'\n||w_th.T @ w - H_pi||**2.: {np.linalg.norm(w_th.T @ w - H_pi) ** 2.: .4f}',
      f'\n||w_th.T @ q - H_pi||**2.: {np.linalg.norm(w_th.T @ q - H_pi) ** 2.: .4f}')

print('\n' + 10 * '.' + 'Subspace spanning')
print('Do w, q span the same subspace?')
cos_thetas = subspace_min_cosine(w, q, tol=1e-2)
print('min', cos_thetas.min().round(3), 'mean', cos_thetas.mean().round(3), 'std', cos_thetas.std().round(3))

print('\nDo w, w_th span the same subspace?')
cos_thetas = subspace_min_cosine(w, w_th, tol=1e-2)
print('min', cos_thetas.min().round(3), 'mean', cos_thetas.mean().round(3), 'std', cos_thetas.std().round(3))

print('\nDo q, w_th span the same subspace?')
cos_thetas = subspace_min_cosine(q, w_th, tol=1e-2)
print('min', cos_thetas.min().round(3), 'mean', cos_thetas.mean().round(3), 'std', cos_thetas.std().round(3))

print('\n' + 10 * '.' + 'Approximations by W_th')
print('how close the approximation of w by w_th?') 
w_on_th = orthogonal_procrustes(w, w_th)

err = np.linalg.norm(w - w_on_th @ w_th, 'fro')
print(f"||w - w_on_th @ w_th|| {err.round(3)}")

rel_err = (
    np.linalg.norm(w - w_on_th @ w_th, 'fro')
    / np.linalg.norm(w_th, 'fro')
)
print(f"||w - w_on_th @ w_th|| / ||w_th|| {rel_err.round(3)}")

print('\nhow close the approximation of q by w_th?') 
q_on_th = orthogonal_procrustes(q, w_th)

err = np.linalg.norm(q - q_on_th @ w_th, 'fro')
print(f"||q - q_on_th @ w_th|| {err.round(3)}")

rel_err = (
    np.linalg.norm(q - q_on_th @ w_th, 'fro')
    / np.linalg.norm(w_th, 'fro')
)
print(f"||q - q_on_th @ w_th|| / ||w_th|| {rel_err.round(3)}")

print('\n' + 10 * '.' + 'Train, test accuracies')
maj_acc_train, min_acc_train = majority_minority_accuracy(
            model_out,
            targs_out,
            pr
        )
        
print(f'..... final train majority acc. {maj_acc_train:.4f}')
print(f'..... final train minority acc. {min_acc_train:.4f}',
      flush=True)

#test outputs
file_path = f"{results_dir.rstrip('/')}/{file_str}_test_results.joblib"
print(f"\nLoading test results from: {file_path}")

targs_out_ts, model_out_ts, ll_ts, _ = joblib.load(file_path)
print(targs_out.shape, targs_out_ts.shape)
maj_acc_test, min_acc_test = majority_minority_accuracy(
            model_out_ts,
            targs_out_ts,
            pr
        )
print(f'..... final test majority acc. {maj_acc_test:.4f}')
print(f'..... final test minority acc. {min_acc_test:.4f}',
      flush=True)

sys.exit(1)

#checking funny things
#funny way of computing class probs? Better to use prob function?
_, counts = np.unique(
    targs_out,
    return_counts=True,
    axis=0
)

#assumes all majority and minority classes have same number of elements;
#doesn't use probs above, as frac is not passed
class_prob = np.sort(
    counts / counts.sum()
)[::-1]
    
print('intermediate class_prob', class_prob, pr)
        


#test accuracies

#for i in range(x_lhl.shape[0]):
#        l_n = [np.linalg.norm(x_lhl[i] - ccm.T[j]) for j in range(C)]
#        y_ncc_pred[i] = np.argmin(np.array(l_n))
#    
#    print("conf m of nearest class center prediction\n", 
#           confusion_matrix(y, y_ncc_pred))
#    print("acc of nearest class center prediction", 
#           accuracy_score(y, y_ncc_pred))
#    print("coincidence of model and ncc predictions", 
#           accuracy_score(class_pred, y_ncc_pred))


print(np.linalg.norm(w - w_on_th @ w_th))



# train losses
unos = np.ones(ll.shape[0])
w_pred_train = ll @ w + unos.reshape(-1, 1) @ b.reshape(1, -1)
q_pred_train = ll @ q + unos.reshape(-1, 1) @ b.reshape(1, -1)

# w, ccm reduced loss
H_pi = h_pi(pr)
w_w_pred = w.T @ w + b.reshape(-1, 1) @ pr.reshape(1, -1)
w_q_pred = w.T @ q + b.reshape(-1, 1) @ pr.reshape(1, -1)
q_q_pred = q.T @ q + b.reshape(-1, 1) @ pr.reshape(1, -1)

w_theo_w_theo_pred = w_opt.T @ w_opt + b.reshape(-1, 1) @ pr.reshape(1, -1)
w_theo_w_pred = w_opt.T @ w + b.reshape(-1, 1) @ pr.reshape(1, -1)
w_theo_q_pred = w_opt.T @ q + b.reshape(-1, 1) @ pr.reshape(1, -1)

print('\n' + 10 * '.' + 'train, w, ccm mse')
print(f'mean ||w_pred_train - targs_out||**2.: {np.linalg.norm(w_pred_train - targs_out) ** 2. / ll.shape[0]: .4f}')
print(f'mean ||q_pred_train - targs_out||**2.: {np.linalg.norm(q_pred_train - targs_out) ** 2. / ll.shape[0]: .4f}')

print(f'\n||w.T @ w + b - H_pi||**2.: {np.linalg.norm(w_w_pred - H_pi) ** 2.: .4f}',  
      f'\n||w.T @ q + b  - H_pi||**2.: {np.linalg.norm(w_q_pred - H_pi) ** 2.: .4f}',
      f'\n||q.T @ q + b - H_pi||**2.: {np.linalg.norm(q_q_pred - H_pi) ** 2.: .4f}') 

print(f'\n||w_opt.T @ w_opt + b - H_pi||**2.: {np.linalg.norm(w_theo_w_theo_pred - H_pi) ** 2.: .4f}', 
      f'\n||w_opt.T @ w + b - H_pi||**2.: {np.linalg.norm(w_theo_w_pred - H_pi) ** 2.: .4f}', 
      f'\n||w_opt.T @ q + b - H_pi||**2.: {np.linalg.norm(w_theo_q_pred - H_pi) ** 2.: .4f}')

print(f'\n||w.T @ w - H_pi||**2.: {np.linalg.norm(w.T @ w - H_pi) ** 2.: .4f}',  
      f'\n||w.T @ q - H_pi||**2.: {np.linalg.norm(w.T @ q - H_pi) ** 2.: .4f}',
      f'\n||q.T @ q - H_pi||**2.: {np.linalg.norm(q.T @ q - H_pi) ** 2.: .4f}') 

# Final w, b norms
print('\n' + 10 * '.' + 'q, w, q-w, q-w_theo, w-w_theo, b norms')
print(f'q: {np.linalg.norm(q): .4f}', 
      f'w: {np.linalg.norm(w): .4f}',  
      f'q-w: {np.linalg.norm(q-w): .4f}', 
      f'q-w_opt: {np.linalg.norm(q-w_opt): .4f}',  
      f'w-w_opt: {np.linalg.norm(w-w_opt): .4f}', 
      f'b: {np.linalg.norm(b): .4f}')

#print('\n' + 10 * '.' + f'q @ w_opt mean: {np.abs(q.T.dot(w_opt)).mean(): .4f} std: {np.abs(q.T.dot(w_opt)).std(): .4f}')

# Final anchored loss
print('\n' + 10 * '.' + 'final anchor loss')
print(f'||w - w_opt.T||**2. / w.shape[0] + ||b||**2.: {np.linalg.norm(w - w_opt) ** 2. / w_opt.shape[0] + np.linalg.norm(b) ** 2.: .4f}')

# Theoretical cos, norm values
cos_column = (np.sqrt(pr) / np.sqrt(1. - pr)).reshape(-1, 1)
theo_cos_matrix = -cos_column @ cos_column.T
theo_cos_matrix += np.eye(n_classes) + np.diag(pr / (1 - pr))

theo_w_norms = np.sqrt(1. - pr)
theo_ccm_norms = np.sqrt((1. - pr) / pr)

# NC1
print('\n' + 10 * '.' + ' traces')
s_with = ff.s_within(ll, label) / ll.shape[0]
s_betw = ff.s_between(ll, label) / ll.shape[0]
nc1_tr = np.trace(np.linalg.pinv(s_betw, rcond=1.e-5) @ s_with) / n_classes
print(f'trace (s_B^+ s_W) / C: {nc1_tr: .6f}')

# NC2 norms
print('\n' + 10 * '.' + ' NC2 norms')
nw = np.linalg.norm(w, axis=0)
nh = np.linalg.norm(ccm, axis=0)
nq = np.linalg.norm(q, axis=0)

print(f'    ||w|| - ||w_teo||: max {np.abs(nw - theo_w_norms).max(): 8.4f} \tmean {np.abs(nw - theo_w_norms).mean(): 8.4f}')
print(f'||ccm|| - ||ccm_teo||: max {np.abs(nh - theo_ccm_norms).max(): 8.4f} \tmean {np.abs(nh - theo_ccm_norms).mean(): 8.4f}')
print(f'    ||q|| - ||q_teo||: max {np.abs(nq - theo_w_norms).max(): 8.4f} \tmean {np.abs(nq - theo_w_norms).mean(): 8.4f}')

# NC2 cosines
print('\n' + 10 * '.' + ' NC2 cosines')
ww = w.T @ w
hh = ccm.T @ ccm
qq = q.T @ q

normw = nw.reshape(-1, 1) @ nw.reshape(-1, 1).T
normh = nh.reshape(-1, 1) @ nh.reshape(-1, 1).T
normq = nq.reshape(-1, 1) @ nq.reshape(-1, 1).T

cos_w = ww / normw
cos_h = hh / normh
cos_q = qq / normq

print(f'cos_w - cos_theo max {np.abs(cos_w - theo_cos_matrix).max().round(3)} \tmean {np.abs(cos_w - theo_cos_matrix).mean().round(3)}')
print(f'cos_h - cos_theo max {np.abs(cos_h - theo_cos_matrix).max().round(3)} \tmean {np.abs(cos_h - theo_cos_matrix).mean().round(3)}')
print(f'cos_q - cos_theo max {np.abs(cos_q - theo_cos_matrix).max().round(3)} \tmean {np.abs(cos_q - theo_cos_matrix).mean().round(3)}')

#print('max, mean h cos diff', np.abs(cos_h - theo_cos_matrix).max().round(3), np.abs(cos_h - theo_cos_matrix).mean().round(3))
#print('max, mean q cos diff', np.abs(cos_q - theo_cos_matrix).max().round(3), np.abs(cos_q - theo_cos_matrix).mean().round(3))
                
# NC3: w, q collinearity  
print('\n' + 10 * '.' + ' NC3 n, w collinearity')     
n_w_ccm_dif = ccm / np.linalg.norm(ccm, axis=0) - w / np.linalg.norm(w, axis=0)
n_w_q_dif = q / np.linalg.norm(q, axis=0) - w / np.linalg.norm(w, axis=0)

#print(f'collinearity normalized diff: max {np.abs(n_w_ccm_dif).max()} \tmean {np.abs(n_w_ccm_dif).mean()}')
print('normalized w, ccm norm diffs\n', np.linalg.norm(n_w_ccm_dif, axis=0).round(3))
print('  normalized w, q norm diffs\n', np.linalg.norm(n_w_q_dif, axis=0).round(3))

# Final w SVD
uw, sw, vwT = np.linalg.svd(w)
uq, sq, vqT = np.linalg.svd(q)
uo, so, voT = np.linalg.svd(w_opt)

uw = uw[:, :10]
uq = uq[:, :10]
uo = uo[:, :10]

print('\n' + 10 * '.' + 'SVs of w, q')
print('w:', sw.round(2))
print('q:', sq.round(2))

print('mean abs diff uw vs uq:', np.abs(uw - uq).mean().round(3), '\tvwT vs vqT:', np.abs(vwT - vqT).mean().round(3))
print('mean abs diff uw vs uo:', np.abs(uw - uo).mean().round(3), '\tvwT vs voT:', np.abs(vwT - voT).mean().round(3))
print('mean abs diff uq vs uo:', np.abs(uq - uo).mean().round(3), '\tvqT vs voT:', np.abs(vqT - voT).mean().round(3))

print('\n' + 10 * '.' + 'orthogonal proj p_W(Q)')
p_w_opt_q = project_Q_2_W(q, w_opt)
po_w_opt_q = q - p_w_opt_q

p_w_q = project_Q_2_W(q, w)
po_w_q = q - p_w_q

p_w_opt_w = project_Q_2_W(w, w_opt)
po_w_opt_w = w - p_w_opt_w

p_q_opt_w = project_Q_2_W(q, w_opt)

#print('  are W and Q - p_W(Q) orth?', np.allclose(w.T @ po_w_q, np.zeros((q.shape[1], q.shape[1])), atol=1.e-4))
#print('    are W and p_W(Q) close?', np.allclose(w, p_w_q, atol=1.e-2))
#
#print('\n  are W_opt and Q - p_W_opt(Q) orth?', np.allclose(w_opt.T @ po_w_opt_q, np.zeros((q.shape[1], q.shape[1])), atol=1.e-4))
#print('    are W_opt and p_W_opt(Q) close?', np.allclose(w_opt, p_w_opt_q, atol=1.e-2))
#
#print('\n  are W_opt and W - p_W_opt(W) orth?', np.allclose(w_opt.T @ po_w_opt_w, np.zeros((q.shape[1], q.shape[1])), atol=1.e-4))
#print('    are W_opt and p_W_opt(W) close?', np.allclose(w_opt, p_w_opt_w, atol=1.e-2))

#print('\ncosines of projected w')
#wowo = p_w_opt_w.T @ p_w_opt_w
#nwo = np.linalg.norm(p_w_opt_w, axis=0)
#normwo = nwo.reshape(-1, 1) @ nwo.reshape(-1, 1).T
#
#cos_wo = wowo / normwo
##print(cos_wo)
#print('max, mean p_w cos - theo cos', np.abs(cos_wo - theo_cos_matrix).max().round(3), np.abs(cos_wo - theo_cos_matrix).mean().round(3))

print('\nProjected w, q vs H_pi')
print(f'\npw.T @ w - H_pi: \tmax abs diff {np.abs(p_w_opt_w.T @ w - H_pi).max().round(3)} \tmean abs diff {np.abs(p_w_opt_w.T @ w - H_pi).mean().round(3)}')
print(f'pw.T @ pw - H_pi: \tmax abs diff {np.abs(p_w_opt_w.T @ p_w_opt_w - H_pi).max().round(3)} \tmean abs diff {np.abs(p_w_opt_w.T @ p_w_opt_w - H_pi).mean().round(3)}')
print(f'pw.T @ q - H_pi: \tmax abs diff {np.abs(p_w_opt_w.T @ q - H_pi).max().round(3)} \tmean abs diff {np.abs(p_q_opt_w.T @ q - H_pi).mean().round(3)}')
print(f'pw.T @ pq - H_pi: \tmax abs diff {np.abs(p_w_opt_w.T @ p_q_opt_w - H_pi).max().round(3)} \tmean abs diff {np.abs(p_q_opt_w.T @ p_q_opt_w - H_pi).mean().round(3)}')
print(f'q.T @ q - H_pi: \tmax abs diff {np.abs(p_q_opt_w.T @ p_q_opt_w - H_pi).max().round(3)} \tmean abs diff {np.abs(q.T @ q - H_pi).mean().round(3)}')

print(f'\n||p_w.T @ p_w - H_pi||**2.: {np.linalg.norm(p_w_opt_w.T @ p_w_opt_w - H_pi) ** 2.: .4f}',  
      f'\n||p_w.T @ q - H_pi||**2.: {np.linalg.norm(p_w_opt_w.T @ q - H_pi) ** 2.: .4f}',
      f'\n||p_w.T @ p_q - H_pi||**2.: {np.linalg.norm(p_w_opt_w.T @ p_q_opt_w - H_pi) ** 2.: .4f}',
      f'\n||p_q.T @ p_q - H_pi||**2.: {np.linalg.norm(p_q_opt_w.T @ p_q_opt_w - H_pi) ** 2.: .4f}') 



#print('same_subspace_residual', same_subspace_residual(w, w_opt, tol=1e-2))
print('\nDo w, w_opt span the same subspace?')
cos_thetas = subspace_min_cosine(w, w_opt, tol=1e-2)
print('min', cos_thetas.min().round(3), 'mean', cos_thetas.mean().round(3), 'std', cos_thetas.std().round(3))

print('\nDo q, w_opt span the same subspace?')
cos_thetas = subspace_min_cosine(q, w_opt, tol=1e-2)
print('min', cos_thetas.min().round(3), 'mean', cos_thetas.mean().round(3), 'std', cos_thetas.std().round(3))

print('\nDo q, w span the same subspace?')
cos_thetas = subspace_min_cosine(q, w, tol=1e-2)
print('min', cos_thetas.min().round(3), 'mean', cos_thetas.mean().round(3), 'std', cos_thetas.std().round(3))

print('\nDo p_wt(q), w span the same subspace?')
cos_thetas = subspace_min_cosine(p_q_opt_w, w, tol=1e-2)
print('min', cos_thetas.min().round(3), 'mean', cos_thetas.mean().round(3), 'std', cos_thetas.std().round(3))

#.....................................................
print('\nhow close the projections of w and q to w_opt?') 
A = project_Q_2_W(w, w_opt)
R = orthogonal_procrustes(A, w_opt)

print("norm distances")
err = np.linalg.norm(A - R @ w_opt, 'fro')
print(f"||p(w) - R @ w_opt|| {err.round(3)}")

rel_err = (
    np.linalg.norm(A - R @ w_opt, 'fro')
    / np.linalg.norm(w_opt, 'fro')
)
print(f"||p(w) - R @ w_opt|| / ||w_opt|| {rel_err.round(3)}")

A = project_Q_2_W(q, w_opt)
R = orthogonal_procrustes(A, w_opt)

err = np.linalg.norm(A - R @ w_opt, 'fro')
print(f"\n||p(q) - R @ w_opt|| {err}")

rel_err = (
    np.linalg.norm(A - R @ w_opt, 'fro')
    / np.linalg.norm(w_opt, 'fro')
)
print(f"||p(q) - R @ w_opt|| / ||w_opt|| {rel_err}")


#print(w.T @ q - w.T @ project_Q_2_W(q, w))
A = project_Q_2_W(w, w_opt)
#print(A.T @ q - A.T @ project_Q_2_W(q, w))

print("\nmax, mean std for p_wopt(w).T @ q - p_wopt(w).T @ p_w(q)")
print(np.abs(A.T @ q - A.T @ project_Q_2_W(q, w)).max().round(3),
      #np.abs(w.T @ q - w.T @ project_Q_2_W(q, w)).max(),
      np.abs(A.T @ q - A.T @ project_Q_2_W(q, w)).mean().round(3),
      np.abs(A.T @ q - A.T @ project_Q_2_W(q, w)).std().round(3))

print("\nmax, mean std for wopt.T @ q - wopt.T @ p_w(q)")
print(np.abs(w_opt.T @ q - w_opt.T @ project_Q_2_W(q, w)).max().round(3),
      np.abs(w_opt.T @ q - w_opt.T @ project_Q_2_W(q, w)).mean().round(3),
      np.abs(w_opt.T @ q - w_opt.T @ project_Q_2_W(q, w)).std().round(3))
      
print("\nbehavior of p_w(q):")
#add cos(p_w(q), cos_theo)
q_w = project_Q_2_W(q, w)
nq_w = np.linalg.norm(q_w, axis=0)
print(f'    ||q_w|| - ||w_teo||: max {np.abs(nq_w - theo_w_norms).max(): 8.4f} \tmean {np.abs(nq_w - theo_w_norms).mean(): 8.4f}')
#print(f'    ||q_w|| - ||w_teo||: max {np.abs(nq_w.reshape(-1, 1) - theo_w_norms.reshape(-1, 1)).max(): 8.4f} \tmean {np.abs(nq_w.reshape(-1, 1) - theo_w_norms.reshape(-1, 1)).mean(): 8.4f}')

cos_q_w = (q_w.T @ q_w) / (nq_w.reshape(-1, 1) @ nq_w.reshape(1, -1))

print('|cos_q_w - cos_theo|: \tmax', np.abs(cos_q_w - theo_cos_matrix).max().round(3), 
      '\tmean', np.abs(cos_q_w - theo_cos_matrix).mean().round(3)) 