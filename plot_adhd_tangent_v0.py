"""
Measuring connectivity in tangent space of symmetric positive definite matrices
===============================================================================

This example shows how to extract signals from regions defined by an atlas,
and to estimate different connectivity measures based on these signals.
"""

scatter_pairs = [(0, 1), (2, 3), (20, 30)]
# pairs of regions whose connectivity to be
                                  # scatter plotted


import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from connectivity import vec_to_sym

# Copied from matplotlib 1.2.0 for matplotlib 0.99 compatibility.
_bwr_data = ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0))
plt.cm.register_cmap(cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
    "bwr", _bwr_data))


def plot_matrix(mean_conn, title):
    """Plot connectivity matrix, for a given measure. """

    mean_conn = mean_conn.copy()  # avoid side effects

    # Compute maximum value
    vmax = np.abs(mean_conn).max()
    if vmax <= 2e-16:
        vmax = 0.1

    # Display connectivity matrix
    plt.figure()
    plt.imshow(mean_conn, interpolation="nearest",
              vmin=-vmax, vmax=vmax, cmap=plt.cm.get_cmap("bwr"))
    plt.colorbar()
    plt.title("%s " % title)


def scatterplot_matrix(data, names, data2=None, title='Scatterplot Matrix', **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""
    numvars, numdata = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i, j), (j, i)]:
            p1, = axes[x, y].plot(data[x], data[y], **kwargs)
            plt.legend([p1], [title])
            if data2 is not None:
                p2, = axes[x, y].plot(data2[x], data2[y], linestyle='none',
                                      marker='x', color='red', mfc='none')
                plt.legend([p1, p2], [title, 'reference'])

    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    import itertools
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

    fig.suptitle(title)


def plot_scatter(conns, pairs, title):
    """Scatter plots of connectivity coefficients for pairs of regions. """

    conns = conns.copy()  # avoid side effects
    n_pairs = len(pairs)
    n_plots = n_pairs * (n_pairs - 1) / 2

    plt.figure()
    for j, pair1 in enumerate(pairs):
        for k in xrange(j + 1, n_pairs):
            plt.subplot(1, n_plots,  j * n_plots + k - j - 1)
            plt.scatter(vec_to_sym(conns)[:, pair1[0], pair1[-1]],
                        vec_to_sym(conns)[:, pairs[k][0], pairs[k][-1]])
            plt.xlabel(pair1)
            plt.ylabel(pairs[k])
            plt.axis('equal')

    plt.title(title)


print("-- Fetching datasets ...")
import nilearn.datasets
# Number of subjects to consider for connectivity computations
n_subjects = 40
atlas = nilearn.datasets.fetch_msdl_atlas()
dataset = nilearn.datasets.fetch_adhd(n_subjects=n_subjects)

import nilearn.image
import nilearn.input_data

import joblib
mem = joblib.Memory(".")

subjects = []

for subject_n in range(n_subjects):
    filename = dataset["func"][subject_n]
    print("Processing file %s" % filename)

    print("-- Computing confounds ...")
    confound_file = dataset["confounds"][subject_n]
    hv_confounds = mem.cache(nilearn.image.high_variance_confounds)(filename)

    print("-- Computing region signals ...")
    masker = nilearn.input_data.NiftiMapsMasker(
        atlas["maps"], resampling_target="maps", detrend=True,
        low_pass=None, high_pass=0.01, t_r=2.5, standardize=True,
        memory=mem, memory_level=1, verbose=1)
    region_ts = masker.fit_transform(filename,
                                     confounds=[hv_confounds, confound_file])
    subjects.append(region_ts)


print("-- Measuring connecivity by covariance ...")
from connectivity import CovEmbedding
estimator = {'kind': None, 'cov_estimator': None}
cov_embedding = CovEmbedding(**estimator)
covariances = cov_embedding.fit_transform(subjects)

print("-- Measuring connecivity by correlation ...")
estimator = {'kind': 'correlation', 'cov_estimator': None}
cov_embedding = CovEmbedding(**estimator)
correlations = cov_embedding.fit_transform(subjects)

print("-- Measuring connecivity by partial correlation ...")
estimator = {'kind': 'partial correlation', 'cov_estimator': None}
cov_embedding = CovEmbedding(**estimator)
partial_correlations = cov_embedding.fit_transform(subjects)

print("-- Measuring connecivity in tangent space...")
estimator = {'kind': 'tangent', 'cov_estimator': None}
cov_embedding = CovEmbedding(**estimator)
tangents = cov_embedding.fit_transform(subjects)

print("-- Displaying results")
title = "correlation coefficients for region pairs {}".format(scatter_pairs)
plot_matrix(vec_to_sym(correlations.mean(axis=0)), "correlations")

title = "covariance coefficients for region pairs {}".format(scatter_pairs)
plot_matrix(vec_to_sym(covariances.mean(axis=0)), "covariances")
m0 = vec_to_sym(covariances)[:,3:5,5:7]
scatterplot_matrix(m0.reshape(4, 40), ['L', 'med', 'front', 'R'], title="covariances",
                   linestyle='none', marker='o', color='black', mfc='none')
plt.figure()
plt.hist(vec_to_sym(covariances)[:,0,1], 30, normed=True)
plt.title('covariances')

title = "partial correlation coefficients for region pairs {}".format(
    scatter_pairs)
plot_matrix(vec_to_sym(partial_correlations.mean(axis=0)),
            "partial correlations")
m = vec_to_sym(partial_correlations)[:,3:5,5:7]
scatterplot_matrix(m.reshape(4, 40), ['L', 'med', 'front', 'R'], title="partial correlations",
                   linestyle='none', marker='o', color='black', mfc='none')
plt.figure()
plt.hist(vec_to_sym(partial_correlations)[:,0,1], 30, normed=True)
plt.title('partial correlations')

title = "tangent coefficients for region pairs {}".format(scatter_pairs)
plot_matrix(cov_embedding.mean_cov_, "covariances geometric mean")
m = vec_to_sym(tangents)[:,3:5,5:7]
scatterplot_matrix(m.reshape(4, 40), ['L', 'med', 'front', 'R'],
                   data2=m0.reshape(4, 40), title="tangent",
                   linestyle='none', marker='o', color='black', mfc='none')
plt.figure()
plt.hist(vec_to_sym(tangents)[:,0,1], 30, normed=True)
plt.title('tangent')
plt.show()