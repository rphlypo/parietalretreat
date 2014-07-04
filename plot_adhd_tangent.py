"""
Measuring connectivity in tangent space of symmetric positive definite matrices
===============================================================================

This example shows how to extract signals from regions defined by an atlas,
and to estimate different connectivity measures based on these signals.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

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


def scatterplot_matrix(coefs, coefs_ref, n_subjects, names,
                       title='measure', title_ref='reference measure'):
    """Plots a scatterplot matrix of subplots. Each connectivity coefficient is
    scatter plotted for a given measure against a reference measure."""
    coefs = coefs.copy()
    coefs = coefs.reshape(-1, n_subjects)
    coefs_ref = coefs_ref.copy()
    coefs_ref = coefs_ref.reshape(-1, n_subjects)
    n_coefs, _ = coefs.shape
    fig, axes = plt.subplots(nrows=n_coefs, ncols=n_coefs, figsize=(8, 8))
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

    # Plot the coefs.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i, j), (j, i)]:
            p1, = axes[x, y].plot(coefs[x], coefs[y], linestyle='none',
                                  marker='o', color='red', mfc='none')
            p2, = axes[x, y].plot(coefs_ref[x], coefs_ref[y], linestyle='none',
                                  marker='x', color='blue', mfc='none')
            plt.legend([p1, p2], [title, title_ref])

    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i, i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    import itertools
    for i, j in zip(range(n_coefs), itertools.cycle((-1, 0))):
        axes[j, i].xaxis.set_visible(True)
        axes[i, j].yaxis.set_visible(True)

    fig.suptitle(title + ' vs ' + title_ref)


def plot_scatter(conns, pairs, title):
    """Scatter plots of connectivity coefficients for pairs of regions. """


print("-- Fetching datasets ...")
import nilearn.datasets

atlas = nilearn.datasets.fetch_msdl_atlas()
dataset = nilearn.datasets.fetch_adhd()

import nilearn.image
import nilearn.input_data

import joblib
mem = joblib.Memory(".")

# Number of subjects to consider for connectivity computations
n_subjects = 40
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
from connectivity import CovEmbedding, vec_to_sym

estimator = {'kind': None, 'cov_estimator': None}
cov_embedding = CovEmbedding(**estimator)
covariances = vec_to_sym(cov_embedding.fit_transform(subjects))

print("-- Measuring connecivity by correlation ...")
estimator = {'kind': 'correlation', 'cov_estimator': None}
cov_embedding = CovEmbedding(**estimator)
correlations = vec_to_sym(cov_embedding.fit_transform(subjects))

print("-- Measuring connecivity by partial correlation ...")
estimator = {'kind': 'partial correlation', 'cov_estimator': None}
cov_embedding = CovEmbedding(**estimator)
partial_correlations = vec_to_sym(cov_embedding.fit_transform(subjects))

print("-- Measuring connecivity in tangent space...")
estimator = {'kind': 'tangent', 'cov_estimator': None}
cov_embedding = CovEmbedding(**estimator)
tangents = vec_to_sym(cov_embedding.fit_transform(subjects))

print("-- Displaying results")
plot_matrix(covariances.mean(axis=0), "covariances arithmetic mean")
coefs_ref = covariances[:, 3:5, 5:7]

plot_matrix(partial_correlations.mean(axis=0),
            "partial correlations mean")
coefs = partial_correlations[:, 3:5, 5:7]
scatterplot_matrix(coefs, coefs_ref, n_subjects,
                   ['L DMN', 'med DMN', 'front DMN', 'R DMN'],
                   title="partial correlation", title_ref='covariance')

plot_matrix(cov_embedding.mean_cov_, "covariances geometric mean")
coefs = tangents[:, 3:5, 5:7]
scatterplot_matrix(coefs, coefs_ref, n_subjects,
                   ['L DMN', 'med DMN', 'front DMN', 'R DMN'],
                   title="tangent", title_ref='covariance')
plt.show()