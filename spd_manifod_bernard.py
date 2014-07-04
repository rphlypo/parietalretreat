# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 17:06:04 2014

@author: bernardng
"""

def log_map(x, displacement, mean=False):
    """ The Riemannian log map at point 'displacement'.
If several points are given, the mean is returned.

See algorithm 2 of Fletcher and Joshi, Sig Proc 87 (2007) 250
"""
    #x = np.asanyarray(x)
    vals, vecs, success_flag = my_eigh(displacement)
    sqrt_vals = np.sqrt(vals)
    whitening = (vecs/sqrt_vals).T
    if len(x.shape) == 2:
        vals_y, vecs_y, success_flag = my_eigh(np.dot(np.dot(whitening, x),
                                                whitening.T))
        sqrt_displacement = np.dot(vecs*sqrt_vals, vecs_y)
        return np.dot(sqrt_displacement*np.log(vals_y), sqrt_displacement.T)
    sqrt_displacement = vecs*sqrt_vals
    y = list()
    for this_x in x:
        vals_y, vecs_y, success_flag = my_eigh(np.dot(
                                                np.dot(whitening, this_x),
                                                whitening.T))
        y.append(np.dot(vecs_y*np.log(vals_y), vecs_y.T))
    y = my_stack(y)
    if mean:
        y = np.mean(y, axis=0)
        return np.dot(np.dot(sqrt_displacement, y), sqrt_displacement.T)
    return my_stack([np.dot(np.dot(sqrt_displacement, this_y),
                                sqrt_displacement.T)
                     for this_y in y])
    

def exp_map(x, displacement):
    """ The Riemannian exp map at point 'displacement'.

See algorithm 1 of Fletcher and Joshi, Sig Proc 87 (2007) 250
"""
    vals, vecs, success_flag = my_eigh(displacement)
    sqrt_vals = np.sqrt(vals)
    whitening = (vecs/sqrt_vals).T
    vals_y, vecs_y, success_flag = my_eigh(np.dot(np.dot(whitening, x),
                                            whitening.T))
    sqrt_displacement = np.dot(vecs*sqrt_vals, vecs_y)
    return np.dot(sqrt_displacement*np.exp(vals_y), sqrt_displacement.T)


def log_mean(population_covs, eps=1e-5):
    """ Find the Riemannien mean of the the covariances.

See algorithm 3 of Fletcher and Joshi, Sig Proc 87 (2007) 250
"""
    step = 1.
    mean = population_covs[0]
    N = mean.size
    eps = N*eps
    direction = old_direction = log_map(population_covs, mean, mean=True)
    while frobenius(direction) > eps:
        direction = log_map(population_covs, mean, mean=True)
        mean = exp_map(step*direction, mean)
        assert np.all(np.isfinite(direction))
        if frobenius(direction) > frobenius(old_direction):
            step = .8*step
            old_direction = direction
    return mean