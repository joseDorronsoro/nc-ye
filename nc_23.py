#!/usr/bin/env python
# coding: utf-8
########################################################################################################
import numpy as np
import joblib
import fisher_functions as ff

# -------- parameters ---------------------------------------------------------
#params
fr = 0.01

opt = 'adamw'
opt = 'sgd'

if opt == 'sgd':
    lrf = 1.0
elif opt == 'adamw':
    lrf = 0.01

bs = 192
wd = 0.0005
hrd = 0.0005
epochs = 600
warm_epochs = 20
l_anc = 0.01
l_center = 1e-06

#bunch_mnist_ef_ye_0.01_opbslrwd_sgd_192_1.0_0.0005_epwu_600_20_hrlalc_0.0005_0.01_1e-06_0_train_results


#bunch_mnist_ye_0.1_sgd_128_1.0_0.0005_1_0.01_20_0_train_results

# ------- frozen weight matrix ------------------------------------------------
def probs(frac):
    """Imbalanced probabilities
    """
    pr = np.array(5 * [1.] + 5 * [float(frac)])
    pr = pr / pr.sum()
    
    return pr 



def householder_V(pr):
    """Householder V matrix for the SVD of h_pi
    """
    v = (np.sqrt(pr) + np.array(9 * [0.] + [1.])).reshape(-1, 1)
    V = (2. / (v.T @ v)) * (v @ v.T) - np.eye(len(pr))
    
    return V



def h_pi(pr):
    """H_pi matrix from probs.
    
    Good for checking that froW bwloNot used.
    """
    H_pi = np.eye(10) - np.sqrt(pr).reshape(-1, 1) @ np.sqrt(pr ).reshape(-1, 1).T

    return H_pi



def optimal_W(dim, frac, verbose=False):
    """Frozen weight matrix for Y targets
    """
    pr = probs(frac)
    V = householder_V(pr)
    
    J = np.eye(len(pr))
    J[-1, -1] = 0.
    
    U = np.eye(dim)[ : , : len(pr)]
    
    if verbose:
        print('\n' + 10 * '.' + ' checking optimal weight matrix')
        print('U.T @ U:', np.allclose(U.T @ U, np.eye(len(pr))))
        print('V @ V.T:', np.allclose(V @ V.T, np.eye(len(pr))))
        print('diag J', np.diagonal(J))
        print('h_pi svd', np.allclose(V @ J @ V.T, h_pi(pr)), '\n')
        #print(np.linalg.norm(U @ J @ V.T))
    
    return U @ J @ V.T

# -----------------------------------------------------------------------------
# subspace projection function
def  project_Q_2_W(Q, W):
    Q_proj = W @ np.linalg.pinv(W, rcond=1.e-3) @ Q
    return Q_proj
    
    
def same_subspace_residual(A: np.ndarray, B: np.ndarray, tol: float = 1e-5) -> bool:
    """
    Checks if col(A) == col(B) by projecting B onto the column space of A.
    """
    # 1. Compute orthonormal bases via economy SVD
    UA, SA, _ = np.linalg.svd(A, full_matrices=False)
    UB, SB, _ = np.linalg.svd(B, full_matrices=False)
    
    # 2. Determine effective ranks
    rank_A = np.sum(SA > tol)
    rank_B = np.sum(SB > tol)
    
    if rank_A != rank_B:
        return False
    
    UA_k = UA[:, :rank_A]  # Shape: [d, k]
    
    # 3. Project B onto col(A): Proj = UA_k @ UA_k.T @ B
    proj_B = UA_k @ (UA_k.T @ B)
    residual = np.linalg.norm(B - proj_B, ord='fro')
    
    return float(residual) < tol
    
    
def subspace_min_cosine(A: np.ndarray, B: np.ndarray, tol: float = 1e-5) -> float:
    """
    Returns the cosine of the worst-case principal angle between col(A) and col(B).
    Returns 1.0 for identical subspaces, 0.0 for orthogonal/different dimensions.
    """
    print('cos_thetas')
    UA, SA, _ = np.linalg.svd(A, full_matrices=False)
    UB, SB, _ = np.linalg.svd(B, full_matrices=False)
    
    rank_A = np.sum(SA > tol)
    rank_B = np.sum(SB > tol)
    
    #if rank_A != rank_B or rank_A == 0:
    #    return 0.0
     
    common_rank = min(rank_A, rank_B)
    UA_k = UA[:, :common_rank]
    UB_k = UB[:, :common_rank]
    
    # Cosines of principal angles are singular values of UA.T @ UB
    cos_thetas = np.linalg.svd(UA_k.T @ UB_k, compute_uv=False)
    
    #print('cos_thetas', rank_A, cos_thetas)
    return float(np.min(cos_thetas))
    
# -----------------------------------------------------------------------------

#print params
print('\n' + 10 * '.' + 'params')
#print('opt:', opt, '\tfrac:', fr, 
#      '\tlr_fac:', lrf, '\tlambda_anch:', l_anc,
#      '\tepochs:', epochs)

print(
'bs = ', bs,
'\twd = ', wd, 
'\thrd = ', hrd,
'\tepochs = ', epochs, 
'\twarm_epochs = ', warm_epochs, 
'\tl_anc = ', l_anc,
'\tl_center = ', l_center
)



#results_dir = '/mnt/e/ongoing/nn_collapse/torch/results/lr_1.0/mnist_test_frac_' + str(fr) + '/'
results_dir = '../nc-ye_exps/'
#bunch_mnist_ef_ye_0.01_opbslrwd_sgd_192_1.0_0.0005_epwu_600_20_hrlalc_0.0005_0.01_1e-06_0_train_results
file_str = ('bunch_mnist_ef_ye_' + str(fr) + 
            '_opbslrwd_sgd_' + str(bs) + '_' +  str(lrf) + '_' + str(wd) + 
            '_epwu_' + str(epochs) + '_' + str(warm_epochs) +
            '_hrlalc_' + str(hrd) + '_' + str(l_anc) + '_' + str(l_center) + '_0')

targs_out, model_out, ll, w_b_lhl = joblib.load(results_dir + file_str + '_train_results.joblib')

w, b = w_b_lhl
w, b = w.T, b.T

w_opt = optimal_W(w.shape[0], fr)

label = np.argmax(targs_out, axis=1)
ccm  = ff.ccm_array(ll,  label)

pr = probs(fr)
n_classes = len(pr)
q = ccm @ np.diag(np.sqrt(pr))
qp = np.linalg.pinv(q, rcond=1.e-5)

#train losses
unos = np.ones(ll.shape[0])
w_pred_train = ll @ w + unos.reshape(-1, 1) @ b.reshape(1, -1)
q_pred_train = ll @ q + unos.reshape(-1, 1) @ b.reshape(1, -1)

#w, ccm reduced loss
H_pi = h_pi(pr)
w_w_pred = w.T @ w + b.reshape(-1, 1) @ pr.reshape(1, -1)
w_q_pred = w.T @ q + + b.reshape(-1, 1) @ pr.reshape(1, -1)
q_q_pred = q.T @ q + + b.reshape(-1, 1) @ pr.reshape(1, -1)

w_theo_w_theo_pred = w_opt.T @ w_opt + + b.reshape(-1, 1) @ pr.reshape(1, -1)
w_theo_w_pred = w_opt.T @ w + + b.reshape(-1, 1) @ pr.reshape(1, -1)
w_theo_q_pred = w_opt.T @ q + + b.reshape(-1, 1) @ pr.reshape(1, -1)

print('\n' + 10 * '.' + 'train, w,  ccm mse')
print(f'mean ||w_pred_train - targs_out||**2.: {np.linalg.norm(w_pred_train - targs_out) ** 2. / ll.shape[0]: .4f}')
print(f'mean ||q_pred_train - targs_out||**2.:; {np.linalg.norm(q_pred_train - targs_out) ** 2. / ll.shape[0]: .4f}')

print(f'\n||w_opt.T @ w_pred - H_pi||**2.: {np.linalg.norm(w_w_pred - H_pi) ** 2.: .4f}',  
      f'\n||w_opt.T @ q_pred - H_pi||**2.:  {np.linalg.norm(w_q_pred - H_pi) ** 2.: .4f}',
      f'\n||q.T @ q - H_pi||**2.: : {np.linalg.norm(q_q_pred - H_pi) ** 2.: .4f}') 

print(f'\n||w_opt.T @ w_opt - H_pi||**2.: {np.linalg.norm(w_theo_w_theo_pred - H_pi) ** 2.: .4f}', 
      f'\n||w_opt.T @ w - H_pi||**2.: {np.linalg.norm(w_theo_w_pred - H_pi) ** 2.: .4f}', 
      f'\n||w_opt.T @ q - H_pi||**2. {np.linalg.norm(w_theo_q_pred - H_pi) ** 2.: .4f}')
      
#print(np.abs(w.T @ w - w.T @ q).max())

#Final w, b norms
print('\n' + 10 * '.' + 'q, w, q-w, q-w_theo, w-w_theo, b norms')
print(f'q: {np.linalg.norm(q): .4f}', 
      f'w: {np.linalg.norm(w): .4f}',  
      f'q-w: {np.linalg.norm(q-w): .4f}', 
      f'q-w_opt: {np.linalg.norm(q-w_opt): .4f}',  
      f'w-w_opt: {np.linalg.norm(w-w_opt): .4f}', 
      f'b: {np.linalg.norm(b): .4f}')

print('\n' + 10 * '.' + f'q @ w_opt mean: {np.abs(q.T.dot(w_opt)).mean(): .4f} std: {np.abs(q.T.dot(w_opt)).std(): .4f}')

#Final anchored loss
print('\n' + 10 * '.' + 'final anchor loss')
print(f'||w - w_opt.T||**2. / w.shape[0] + ||b||**2.: {np.linalg.norm(w - w_opt) ** 2. / w_opt.shape[0] + np.linalg.norm(b) ** 2.: .4f}')


#Theoretical cos, norm values
cos_column = (np.sqrt(pr) / np.sqrt(1. - pr)).reshape(-1, 1)
theo_cos_matrix = -cos_column @ cos_column.T
theo_cos_matrix += np.eye(n_classes) + np.diag(pr / (1 - pr))

theo_w_norms = np.sqrt(1. - pr)
theo_ccm_norms = np.sqrt((1. - pr) / pr)


#NC1
print('\n' + 10 * '.' + ' traces')
s_with = ff.s_within(ll, label) / ll.shape[0]
s_betw = ff.s_between(ll, label) / ll.shape[0]
nc1_tr = np.trace(np.linalg.pinv(s_betw, rcond=1.e-5) @ s_with) / n_classes
print(f'trace (s_B^+ s_W) / C: {nc1_tr: .6f}')


#NC2 norms
print('\n' + 10 * '.' + ' NC2 norms')
nw = np.linalg.norm(w, axis=0)
nh = np.linalg.norm(ccm, axis=0)
nq = np.linalg.norm(q, axis=0)

print(f'    ||w|| - ||w_teo||: max {np.abs(nw - theo_w_norms).max(): 8.4f} \tmean {np.abs(nw - theo_w_norms).mean(): 8.4f}')
print(f'||ccm|| - ||ccm_teo||: max {np.abs(nh - theo_ccm_norms).max(): 8.4f} \tmean {np.abs(nh - theo_ccm_norms).mean(): 8.4f}')
print(f'    ||q|| - ||q_teo||: max {np.abs(nq - theo_w_norms).max(): 8.4f} \tmean {np.abs(nq - theo_w_norms).mean(): 8.4f}')

#print('relative norm_diff_w', np.abs((nw - theo_w_norms) / theo_w_norms).max(), np.abs((nw - theo_w_norms) / theo_w_norms).mean())
#print('relative norm_diff_ccm', np.abs(nh - theo_ccm_norms).max(), np.abs(nh - theo_ccm_norms).mean())
#print('relative norm_diff_q', np.abs(nq - theo_w_norms).max(), np.abs(nq - theo_w_norms).mean())

#NC2 cosines
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

print('max, mean w cos diff', np.abs(cos_w - theo_cos_matrix).max(), np.abs(cos_w - theo_cos_matrix).mean())
print('max, mean h cos diff', np.abs(cos_h - theo_cos_matrix).max(), np.abs(cos_h - theo_cos_matrix).mean())
print('max, mean q cos diff', np.abs(cos_q - theo_cos_matrix).max(), np.abs(cos_q - theo_cos_matrix).mean())
                
#NC3: w, ccm collinearity  
print('\n' + 10 * '.' + ' NC3 n, w collinearity')     
n_w_ccm_dif = ccm / np.linalg.norm(ccm, axis=0) - w / np.linalg.norm(w, axis=0)

print('collinearity diff', np.abs(n_w_ccm_dif).max(), np.abs(n_w_ccm_dif).mean())


#Final w SVD
uw, sw, vwT = np.linalg.svd(w)
uq, sq, vqT = np.linalg.svd(q)
uo, so, voT = np.linalg.svd(w_opt)

uw = uw[ : , : 10]
uq = uq[ : , : 10]
uo = uo[ : , : 10]

print('\n' + 10 * '.' + 'SVs of w, q')
print('w:', sw.round(2))
print('q:', sq.round(2))

print('mean abs diff uw vs uq:', np.abs(uw - uq).mean(), '\tvwT vs vqT:', np.abs(vwT - vqT).mean())
print('mean abs diff uw vs uo:', np.abs(uw - uo).mean(), '\tvwT vs voT:', np.abs(vwT - voT).mean())
print('mean abs diff uq vs uo:', np.abs(uq - uo).mean(), '\tvqT vs voT:', np.abs(vqT - voT).mean())

#print(uq.T @ uw) # -np.eye(10, 10))
#print(uq.shape)
#q2 = uq @ np.diag(so) @ vqT
#w2 = uw @ np.diag(so) @ vwT
#
#print(np.abs(q2-w).mean(), np.abs(w2-w_opt).mean())

print('\n' + 10 * '.' + 'orthogonal proj p_W(Q)')
p_w_opt_q = project_Q_2_W(q, w_opt)
po_w_opt_q = q - p_w_opt_q

p_w_q = project_Q_2_W(q, w)
po_w_q = q - p_w_q

p_w_opt_w = project_Q_2_W(w, w_opt)
po_w_opt_w = w - p_w_opt_w

print('  are W and Q - p_W(Q) orth?', np.allclose(w.T @ po_w_q, np.zeros((q.shape[1], q.shape[1])), atol=1.e-4))
print('    are W and p_W(Q) close?',  np.allclose(w, p_w_q, atol=1.e-2))
#print('are W_opt and p_W(Q) close?',  np.allclose(w_opt, p_w_q, atol=1.e-2))
#, np.linalg.norm(w - w_opt))

print('\n  are W_opt and Q - p_W_opt(Q) orth?', np.allclose(w_opt.T @ po_w_opt_q, np.zeros((q.shape[1], q.shape[1])), atol=1.e-4))
print('    are W_opt and p_W_opt(Q) close?',  np.allclose(w_opt, p_w_opt_q, atol=1.e-2))
#print('are W_opt and p_W(Q) close?',  np.allclose(w_opt, p_w_q, atol=1.e-2))

print('\n  are W_opt and W - p_W_opt(W) orth?', np.allclose(w_opt.T @ po_w_opt_w, np.zeros((q.shape[1], q.shape[1])), atol=1.e-4))
print('    are W_opt and p_W_opt(W) close?',  np.allclose(w_opt, p_w_opt_w, atol=1.e-2))


print('\nq.T @ q vs H_pi')

print('max abs diff', np.abs(q.T @ q - H_pi).max())

print('\nrounded diff\n', np.round(q.T @ q - H_pi, 1))

print('same_subspace_residual', same_subspace_residual(w, w_opt, tol=1e-2))
print('same_subspace_min_cosine if min_cos near 1:', subspace_min_cosine(w, w_opt, tol=1e-2))
