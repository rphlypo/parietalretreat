import numpy as np
from sklearn import covariance
from nilearn import signal
import pylab as pl
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import ProbabilisticPCA
from sklearn.pipeline import Pipeline
from scipy.linalg.lapack import get_lapack_funcs
from matrix import untri
from sklearn.covariance import EmpiricalCovariance
#from sklearn import clone
#from .data import load


def covariance_matrix(series, gm_index, confounds=None):
    series = load(series)
    if series.shape[1] == 0:
        # Empty serie because region is empty
        return np.zeros((1, 1))

    if confounds is not None and np.ndim(confounds) == 3:
        confounds_ = []
        for c in confounds:
            c = load(c)
            if isinstance(c, basestring) or np.isfinite(c).all():
                confounds_.append(c)
        confounds = confounds_
    series = signal.clean(series, confounds=confounds)
    estimator = covariance.LedoitWolf()
    # Keep only gm regions
    series = series[:, np.array(gm_index)]
    try:
        estimator.fit(series)
        return estimator.covariance_, estimator.precision_
    except Exception as e:
        print e
        return np.eye(series.shape[1]), np.eye(series.shape[1])


def cov_embedding(covariances):
    """Returns for a list of matrices a list of transformed
    matrices in matrix form, with 1. on the diagonal
    """
    ce = CovEmbedding()
    covariances = ce.fit_transform(covariances)  # shape n(n+1)/2
    if covariances is None:
        return None
    # changed from k=1 to k=0 by Salma
    return np.asarray([untri(c, k=0, fill=1.) for c in covariances])


def match_pairs(covariances, regions):
    # sym = regions[::-1]
    pass


def plot_cov(covariance, vmax=None, cmap='RdBu_r', title=None):
    if vmax is None:
        vmax = np.max(np.abs(covariance))
    pl.matshow(covariance, vmin=-vmax, vmax=vmax, cmap=cmap)
    if title is not None:
        pl.title(title)


def plot_covs(path, covariances, precisions, labels):
    """ Print a single matrix of mean covariance and mean precision.

    Parameters:
    -----------
        covariances: list of covariance matrices

        precisions: list of precision matrices

        labels: list of labels
    """
    assert(len(covariances) == len(precisions))
    assert(len(labels) == len(precisions))
    paths = []
    for l in np.unique(labels):
        mean_cov = np.mean(covariances[labels == l], axis=0)
        mean_prec = np.mean(precisions[labels == l], axis=0)
        np.fill_diagonal(mean_prec, 0.)

        plot_cov(mean_cov, vmax=1., title='Covariance label %d' % l)
        pl.colorbar()
        p = (path + '_cov.png') % l
        pl.savefig(p)
        pl.close()
        paths.append(p)

        plot_cov(mean_prec, title='Precision label %d' % l)
        pl.colorbar()
        p = (path + '_prec.png') % l
        pl.savefig(p)
        pl.close()
        paths.append(p)
    return paths


def my_stack(arrays):
    return np.concatenate([a[np.newaxis] for a in arrays])

# Bypass scipy for faster eigh (and dangerous: Nan will kill it)
my_eigh, = get_lapack_funcs(('syevr', ), np.zeros(1))


def inv_sqrtm(mat):
    """ Inverse of matrix square-root, for symetric positive definite matrices.
    """
    vals, vecs, success_flag = my_eigh(mat)
    return np.dot(vecs / np.sqrt(vals), vecs.T)


def inv(mat):
    """ Inverse of matrix, for symetric positive definite matrices.
    """
    vals, vecs, success_flag = my_eigh(mat)
    return np.dot(vecs / vals, vecs.T)


def sym_to_vec(sym):
    """ Returns the lower triangular part of
    sqrt(2) (sym - Id) + (1-sqrt(2)) * diag(sym - Id),

    sqrt(2) * offdiag(sym) + np.diag(np.diag(sym)) - Id

    shape n (n+1) /2, with n = sym.shape[0]

    Parameters
    ==========
    sym: array
    """
    sym = np.sqrt(2) * sym
    # the sqrt(2) factor
    p = sym.shape[-1]
    sym.flat[::p + 1] = sym.flat[::p + 1] / np.sqrt(2)
    mask = np.tril(np.ones(sym.shape[-2:])).astype(np.bool)
    return sym[..., mask]


def vec_to_sym(vec):
    n = vec.size
    # solve p * (p + 1) / 2 = n subj. to p > 0
    # p ** 2 + p - 2n = 0 & p > 0
    # p = - 1 / 2 + sqrt( 1 + 8 * n) / 2
    p = int((np.sqrt(8 * n + 1) - 1.) / 2)
    mask = np.tril(np.ones((p, p))).astype(np.bool)
    sym = np.empty((p, p), dtype=np.float)
    sym[..., mask] = vec
    sym = (sym + sym.T) / np.sqrt(2)  # divide by 2 multiply by sqrt(2)
    sym.flat[::p + 1] = sym.flat[::p + 1] / np.sqrt(2)
    return sym


def cov_to_corr(cov):
    return cov * np.diag(cov) ** (-1. / 2) *\
        (np.diag(cov) ** (-1. / 2))[..., np.newaxis]


class CovEmbedding(BaseEstimator, TransformerMixin):
    """ Tranformer that returns the coefficients on a flat space to
    perform the analysis.
    """

    def __init__(self, base_estimator=None, kind='tangent'):
        self.base_estimator = base_estimator
        self.kind = kind
#        if self.base_estimator == None:
#            self.base_estimator_ = ...
#        else:
#            self.base_estimator_ = clone(base_estimator)

    def fit(self, X, y=None):
        if self.base_estimator is None:
            self.base_estimator_ = EmpiricalCovariance(
                assume_centered=True)
        else:
            self.base_estimator_ = clone(self.base_estimator)

        if self.kind == 'tangent':
            # self.mean_cov = mean_cov = spd_manifold.log_mean(covs)
            # Euclidean mean as an approximation to the geodesic
            covs = [self.base_estimator_.fit(x).covariance_ for x in X]
            covs = my_stack(covs)
            mean_cov = np.mean(covs, axis=0)
            self.whitening_ = inv_sqrtm(mean_cov)
        return self

    def transform(self, X):
        """Apply transform to covariances

        Parameters
        ----------
        covs: list of array
            list of covariance matrices, shape (n_rois, n_rois)

        Returns
        -------
        list of array, transformed covariance matrices,
        shape (n_rois * (n_rois+1)/2,)
        """
        covs = [self.base_estimator_.fit(x).covariance_ for x in X]
        covs = my_stack(covs)
        p = covs.shape[-1]
        if self.kind == 'tangent':
            id_ = np.identity(p)
            covs = [self.whitening_.dot(c.dot(self.whitening_)) - id_
                    for c in covs]
        elif self.kind == 'partial correlation':
            covs = [cov_to_corr(inv(g)) for g in covs]
        elif self.kind == 'correlation':
            covs = [cov_to_corr(g) for g in covs]
        return np.array([sym_to_vec(c) for c in covs])


if __name__ == '__main__':
    KIND = 'partial correlation'
    KIND = 'observation'
    KIND = 'tangent'

    # load the controls
    control_covs = np.load('/home/sb238920/CODE/NSAP/controls.npy')
#    control_covs = np.mean(control_covs, 1)
    n_controls, n_rois, _ = control_covs.shape

    # load the patients
    patient_covs = np.load('/home/sb238920/CODE/NSAP/patients.npy')
#    patient_covs = np.mean(patient_covs, 1)
    n_patients = len(patient_covs)
    patient_nbs = [4, 13, 18, 15, 16, 20, 22, 27, 30, 36]

    # 'test on control and patients'
    n_components = 0
    embedding = CovEmbedding(kind=KIND)
    pca = ProbabilisticPCA(n_components=n_components)
    model = Pipeline((('embedding', embedding),
                      ('pca', pca)))
    control_model = model.fit(control_covs)

    control_fits = control_model.score(control_covs)
    patient_fits = control_model.score(patient_covs)

    patient_fit_cv = np.zeros(n_patients)
    control_fit_cv = list()

    for n in range(n_controls):
        train = [control_covs[i]
                 for i in range(n_controls) if i != n]
        test = control_covs[n]
        control_model.fit(train)
        control_fit_cv.append(control_model.score([test]))
        patient_fit_cv += control_model.score(patient_covs)

    patient_fit_cv /= n_controls

    #pl.rcParams['text.usetex'] = True
    #pl.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
    pl.figure(['tangent', 'observation',
               'partial correlation'].index(KIND),
              figsize=(2.5, 3))
    pl.clf()
    ax = pl.axes([.1, .15, .5, .7])
    pl.boxplot([control_fit_cv, patient_fit_cv], widths=.25)
    pl.plot(1.26 * np.ones(len(control_fit_cv)), control_fit_cv, '+k',
            markeredgewidth=1)
    pl.plot(2.26 * np.ones(len(patient_fits)),
            patient_fit_cv, '+k',
            markeredgewidth=1)
    pl.xticks((1.13, 2.13), ('controls', 'patients'), size=11)
    title = '%s%s \nspace' % (KIND[0].upper(), KIND[1:])
    pl.text(.05, .1, title,
            #'Partial\ncorrelation\nspace',
            transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='bottom',
            size=12)
    #pl.axis([0.7, 2.5, 401, 799])
    pl.xlim(.7, 2.5)
    #pl.ylim(401, 799)
    ax.yaxis.tick_right()
    pl.yticks(size=9)
    pl.ylabel('Log-likelihood', size=12)
    ax.yaxis.set_label_position('right')
    pl.title('N components=%i' % n_components)
    pl.draw()
