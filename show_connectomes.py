import os.path
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xml_et
import numpy as np
from mayavi import mlab
#from nipy.labs.viz_tools import maps_3d
import joblib
# Major scientific library imports
from scipy import stats
from sklearn import cluster
from sklearn import metrics
from matplotlib import mlab as mlab2
# regular expressions
import re

# Neuroimaging library imports
#from nipy.labs import viz3d

#from parietal.learn.covariance.viz3d import plot_graph
#from parietal.learn.covariance.viz import plot_correlation


###############################################################################
# Helper functions

def mlab_save_views(name, fig):
    # Save pics
    fig.scene.parallel_projection = False
    mlab.view(25, 70, 310, (1.3, -16.1, 3.27))
    fig.scene.disable_render = False

    mlab.savefig(name + '_3d.png')

    mlab.view(0, 90)
    fig.scene.parallel_projection = True
    cam = fig.scene.camera
    cam.zoom(1.8)
    mlab.savefig(name + '_3d_side.png')

    mlab.view(-90, 0)
    mlab.savefig(name + '_3d_top.png')


def group_lateral(G, labels):
    """ recompute the connectivity according to lateralization

    This function permutes the adjacency matrix and the labels of the ROIs
    so that the lateralization is emphasized.

    Input arguments:
    ---------------
    G       : np.ndarray of size (p,p)
        the adjacency matrix of the connectivity graph

    labels  : list or tuple of strings of length p
        the labels of the ROIs

    trimmed : boolean, optional
        whether or not the labels are trimmed so as to no longer start with
        the lateral string "left" or "right"
    """
    p = G.shape[0]
    ix = [lix for lix in np.arange(p) if labels[lix][:4].lower() == "left"]
    ix.extend([rix for rix in np.arange(p)
               if labels[rix][:5].lower() == "right"])
    ix.extend([remix for remix in np.arange(p) if (remix not in ix)])
    return G[np.ix_(ix, ix)], [labels[ii] for ii in ix]


def trim_label(labels):
    p = re.compile("(left|right|Left|Right)")
    if not hasattr(labels, '__iter__'):
        labels = [labels]
    return [p.sub(laterality_abbrev, s) for s in labels]


def laterality_abbrev(match):
    return match.group()[0]


def get_regions(atlas_name="HarvardOxford"):
    if atlas_name == "HarvardOxford" or atlas_name == "HarvardOxfordExt":
        regions = [lbl for lbl in get_fsl_region_labels()]
    if atlas_name == "Juelich":
        regions = [lbl for lbl in get_fsl_region_labels(
                   file_name="Juelich.xml")]
    if atlas_name == "HarvardOxfordExt":
        regions.extend([lbl for lbl in get_fsl_region_labels(
            file_name="HarvardOxford-Subcortical.xml")])
    return regions


def get_labels(atlas_name="HarvardOxford"):
    regions = get_regions(atlas_name)
    return [region["label"] for region in regions]


def plot_adjacency(G,
                   atlas_name="HarvardOxford",
                   labels=None,
                   lateralised=True,
                   trim=True,
                   col_map=None,
                   n_clusters=None,
                   plot_figure=True,
                   title=None,
                   vmin=0.,
                   vmax=1.,
                   fig_name=False):
    if col_map == "hot":
        cmap = plt.cm.hot
    elif col_map == "red_blue":
        cmap = plt.cm.RdBu
    elif col_map == "red_blue_r":
        cmap = plt.cm.RdBu_r
    else:
        cmap = plt.cm.hot_r
    p = G.shape[0]
    if labels is None:
        labels = get_labels(atlas_name)
        if lateralised:
            G, labels = group_lateral(G, labels)
        if trim:
            labels = trim_label(abbreviate_labels(labels))
        if n_clusters is None:
            n_clusters, cluster_labels = nb_clusters(G)
        if n_clusters > 1:
            AC = cluster.AgglomerativeClustering(affinity='precomputed',
                                                 compute_full_tree=True,
                                                 linkage='complete',
                                                 n_clusters=n_clusters)
            cluster_labels = AC.fit(1 - G).labels_
            ix = np.argsort(cluster_labels, kind="mergesort")
            G = G[np.ix_(ix, ix)]
            labels = [labels[ii] for ii in ix]
        else:
            cluster_labels = list(np.zeros((p,)))
    else:
        cluster_labels = labels
    if not plot_figure:
        return cluster_labels
    fig = plt.figure()
    if vmin is None and vmax is None:
        vmax = np.max(np.abs(G))
        vmax += 0.05 * (vmax < 0.05)
        vmin = -vmax
    plt.imshow(G, vmin=vmin, vmax=vmax, interpolation='nearest',
               cmap=cmap)
    ax = plt.gca()
    ax.xaxis.set_ticks_position("top")
    plt.xticks(np.arange(p), labels, rotation=70, size=8,
               va="bottom", ha="left")
    plt.yticks(np.arange(p), labels, rotation=20, size=8,
               va="top", ha="right")
    for c in np.arange(n_clusters - 1):
        ref = cluster_labels[cluster_labels <= c].size - 0.5
        plt.axhline(y=ref, xmin=-0.5, xmax=p - 0.5, linewidth=2)
        plt.axvline(x=ref, ymin=-0.5, ymax=p - 0.5, linewidth=2)
    plt.colorbar()
    if title is not None:
        plt.suptitle(title)
    print fig_name
#    plt.show()
    if fig_name:
        fig.savefig(fig_name, bbox_inches='tight')
        print "saved"
        plt.close(fig)
    return cluster_labels


def get_fsl_region_labels(file_name="HarvardOxford-Cortical-Lateralized.xml",
                          label_path="/usr/share/fsl/data/atlases/"):
    """ xml encoded fsl map data are returned in a list of dictionaries
    """
    tree = xml_et.parse(
        os.path.join(os.path.expanduser(label_path), file_name))
    root = tree.getroot()
    items = [item for item in root]
    data = [item for item in items if item.tag == 'data'][0]
    regions_xml = [region for region in data]
    # list of dictionaries with entries:
    #   index
    #   label
    #   x
    #   y
    #   z
    regions = [region_xml.attrib for region_xml in regions_xml]
    for (region_ix, region) in enumerate(regions_xml):
        regions[region_ix]["label"] = region.text

    return regions


def abbreviate_labels(labels):
    lut = acronym_lut()
    for entry in lut.keys():
        p = re.compile("(" + entry + ")")
        labels = [p.sub(lut[entry], s) for s in labels]
    p = re.compile("(left|right|Left|Right)")
    return labels


def acronym_lut():
    # TODO: complete table, see Wang2011PLoSONE &
    # www.thehumanbrain.info/database/nomenclature.php
    return dict({
        "Frontal Pole": "FP",
        "Insular Cortex": "INS",
        "Superior Frontal Gyrus": "F1",
        "Middle Frontal Gyrus": "F2",
        "Inferior Frontal Gyrus\, pars triangularis": "F3t",
        "Inferior Frontal Gyrus\, pars opercularis": "F3o",
        "Precentral Gyrus": "PRG",
        "Temporal Pole": "TP",
        "Superior Temporal Gyrus\, anterior division": "T1a",
        "Superior Temporal Gyrus\, posterior division": "T1b",
        "Middle Temporal Gyrus\, anterior division": "T2a",
        "Middle Temporal Gyrus\, posterior division": "T2p",
        "Middle Temporal Gyrus\, temporooccipital part": "TO2",
        "Inferior Temporal Gyrus\, anterior division": "T3a",
        "Inferior Temporal Gyrus\, posterior division": "T3p",
        "Inferior Temporal Gyrus\, temporooccipital part": "TO3",
        "Postcentral Gyrus": "POG",
        "Superior Parietal Lobule": "SPL",
        "Supramarginal Gyrus\, anterior division": "SGa",
        "Supramarginal Gyrus\, posterior division": "SGp",
        "Angular Gyrus": "AG",
        "Lateral Occipital Cortex\, superior division": "OLs",
        "Lateral Occipital Cortex\, inferior division": "OLi",
        "Intracalcarine Cortex": "CALC",
        "Frontal Medial Cortex": "FMC",
        "Juxtapositional Lobule Cortex " +
        "\(formerly Supplementary Motor Cortex\)": "SMC",
        "Subcallosal Cortex": "SC",
        "Paracingulate Gyrus": "PAC",
        "Cingulate Gyrus, anterior division": "CGa",
        "Cingulate Gyrus, posterior division": "CGp",
        "Precuneous Cortex": "PCN",
        "Cuneal Cortex": "CN",
        "Frontal Orbital Cortex": "FOC",
        "Parahippocampal Gyrus\, anterior division": "PHa",
        "Parahippocampal Gyrus\, posterior division": "PHp",
        "Lingual Gyrus": "LG",
        "Temporal Fusiform Cortex\, anterior division": "TFa",
        "Temporal Fusiform Cortex\, posterior division": "TFp",
        "Temporal Occipital Fusiform Cortex": "TOF",
        "Occipital Fusiform Gyrus": "OF",
        "Frontal Operculum Cortex": "FO",
        "Central Opercular Cortex": "CO",
        "Parietal Operculum Cortex": "PO",
        "Planum Polare": "PP",
        "Heschl's Gyrus \(includes H1 and H2\)": "H1/2",
        "Planum Temporale": "PT",
        "Supracalcarine Cortex": "SCLC",
        "Occipital Pole": "OP"
    })
    return


def nb_clusters(G):
    """number of clusters associated with minimal silhouette value
    """
    p = G.shape[0]
    score = []
    clabels = []
    for k in np.arange(2, p):
        AC = cluster.AgglomerativeClustering(affinity='precomputed',
                                             compute_full_tree=True,
                                             linkage='complete',
                                             n_clusters=k)
        clabels.append(AC.fit(1 - G).labels_)
        score.append(metrics.silhouette_score(G, metric='precomputed',
                                              labels=clabels[-1]))
    return list(np.arange(2, p))[np.argmin(score)], clabels[np.argmin(score)]


def plot_connectivity_graph(Theta=None, atlas_name="HarvardOxford",
                            fig_name=None, partial=None, retain=.1,
                            cluster_labels=None, plot_networks=False):
    """ plot the connectivity graph inside a glass brain
    """
    if Theta is None:
        Theta = np.identity(96, dtype=np.float)
        Theta[10, 0] = .9
        Theta[86, 3] = -.7

    Theta_ = Theta.copy()
    p = Theta_.shape[0]

    # 3D positions of regions + labels
    regions = get_regions(atlas_name)
    X, Y, Z, roi_labels = zip(
        *[(int(region["x"]),
           int(region["y"]),
           int(region["z"]),
           region["label"])
          for region in regions])
    (x, y, z) = map(np.array, (X, Y, Z))

    # 3D glass image of brain
    fig = mlab.figure(bgcolor=(1, 1, 1), size=(900, 769))
    mlab.clf()
    fig.scene.disable_render = True

    # 2mm cortical map of Harvard_Oxford --> voxel to MNI coordinates (mm)
    if atlas_name in {"HarvardOxford", "HarvardOxfordExt"}:
        affine = np.identity(4, dtype=np.float)
        affine[0, 0] = -2.
        affine[1:3, 1:3] = 2. * affine[1:3, 1:3]
        affine[:3, 3] = np.array([90., -126., -72.])
        affine /= 1.1
    elif atlas_name == "Juelich":
        affine = np.identity(4, dtype=np.float)
        affine[0, 0] = -2.
        affine[1:3, 1:3] = 2. * affine[1:3, 1:3]
        affine[:3, 3] = np.array([90., -126., -72.])
        affine /= 1.1
    (x, y, z) = maps_3d.coord_transform(x, y, z, affine)

    partial_var = 1. / Theta_.flat[::p + 1]
    if partial:
        Theta_ = -Theta_.dot(np.diag(partial_var))

    pctl = (1 - retain) * 100
    thr = stats.scoreatpercentile(
        np.abs(Theta_[np.triu_indices(Theta_.shape[0], k=1)]), pctl)
    Theta_[np.abs(Theta_) < thr] = 0

    #rois = mlab.points3d(x, y, z, partial_var, scale_factor=10, figure=fig)
    # construct edges as a list of tuples [(p1,p2),(p3,p4),...] which means
    # that an edge exists between vertices p1 and p2, p3 and p4, and so on
    #Theta_ = np.tril(Theta, -1)
    #edges_n1, edges_n2 = np.nonzero(Theta_)

    if cluster_labels is not None:
        n_clusters = np.unique(cluster_labels).size
        n_members = [cluster_labels[cluster_labels == c].size
                     for c in np.arange(n_clusters - 1)]
        n_large_clusters = 5
        large_cluster_ix = np.argsort(
            n_members, kind="mergesort")[-1:-n_large_clusters - 1:-1]
        n_networks = n_large_clusters + 1
        network = [mlab2.find(large_cluster_ix == cluster_labels[ix])[0]
                   if cluster_labels[ix] in large_cluster_ix
                   else n_large_clusters for ix in np.arange(p)]
    else:
        network = None

    vmin = np.min(Theta_[np.abs(Theta_) != 0])
    vmax = np.max(np.abs(Theta_))
    tubes, nodes = plot_graph(-Theta_, x, y, z,
                              node_size=.6,
                              edge_vmin=vmin,
                              edge_vmax=vmax,
                              node_colormap='spectral',
                              node_color=(0.2, 0.2, 0.2),
                              node_scalar=network,
                              tube_radius=.15)
    #tubes.parent.parent.parent.filter.vary_radius = \
    #    'vary_radius_by_absolute_scalar'

    #tubes.module_manager.scalar_lut_manager.reverse_lut = True
    # Make points of the lut transparent
    lut = tubes.module_manager.scalar_lut_manager.lut.table.to_array()
    lut = 255 * plt.cm.hot_r(np.linspace(0, 1, 256))
    tubes.module_manager.scalar_lut_manager.lut.table = lut
    tubes.update_pipeline()
    nodes.update_pipeline()

    viz3d.plot_anat_3d(outline_color=(0, 0, 0), gyri_opacity=0.15)
    fig.scene.disable_render = False
    if fig_name:
        mlab_save_views(fig_name, fig)
        return fig
    if plot_networks:
        ix_range = np.arange(p)
        for nw_ix in np.arange(n_networks - 1):
            nw_members = ix_range[cluster_labels == large_cluster_ix[nw_ix]]
            fig2 = _plot_subnetwork_graph(
                Theta[np.ix_(nw_members, nw_members)],
                (x[nw_members], y[nw_members], z[nw_members]), nw_ix)
            if fig_name:
                mlab_save_views(fig_name + "_nw%i" % nw_ix, fig2)
    return large_cluster_ix


def _plot_subnetwork_graph(Theta, coords, network_ix):
    (x, y, z) = coords
    p = Theta.shape[0]
    network_ix = network_ix * np.ones((p,))
    # 3D glass image of brain
    fig = mlab.figure(bgcolor=(1, 1, 1), size=(900, 769))
    mlab.clf()
    fig.scene.disable_render = True
    vmin = np.min(Theta[np.abs(Theta) != 0])
    vmax = np.max(np.abs(Theta))
    tubes, nodes = plot_graph(-Theta, x, y, z,
                              node_size=.6,
                              edge_vmin=vmin,
                              edge_vmax=vmax,
                              node_colormap='spectral',
                              node_color=(0.2, 0.2, 0.2),
                              node_scalar=network_ix,
                              tube_radius=.15)
    lut = tubes.module_manager.scalar_lut_manager.lut.table.to_array()
    lut = 255 * plt.cm.hot_r(np.linspace(0, 1, 256))
    tubes.module_manager.scalar_lut_manager.lut.table = lut
    tubes.update_pipeline()
    #nodes.module_manager.scalar_lut_manager.lut.table = color
    nodes.update_pipeline()

    viz3d.plot_anat_3d(outline_color=(0, 0, 0), gyri_opacity=0.15)
    fig.scene.disable_render = False
    return fig


def get_prob_adjacency_graph(path):
    HCP_results = joblib.load(path)
    Supp = [S[..., np.newaxis] for S in HCP_results["supp_set"]]
    return np.concatenate(Supp, axis=2).sum(axis=2) / np.float(len(Supp))


def partial_corr(Theta):
    scale = np.diag(1. / np.diag(Theta))
    return -np.dot(Theta, scale)


def get_atlas_path(atlas_name):
    HOcl = "HarvardOxford-Cortical-Lateralized"
    if atlas_name is None or atlas_name == HOcl:
        return os.path.join("/usr/share/fsl/data/atlases/HarvardOxford/",
                            "HarvardOxford-cortl-prob-2mm.nii.gz")


def evaluate_gen_lik(file_name):
    results = joblib.load(file_name)
    LL = [np.concatenate([s[np.newaxis, ...] for s in results["ll_supp"]]),
          np.concatenate([s[np.newaxis, ...] for s in results["ll_alpha"]]),
          np.concatenate([s[np.newaxis, ...] for s in results["ll_supp_ips"]]),
          np.concatenate([s[np.newaxis, ...] for s in results["ll_alpha_ips"]])
          ]
    LL = np.concatenate([mx[..., np.newaxis] for mx in LL], axis=-1)
    plt.figure()
    plt.plot([400, 800, 1200], LL.mean(axis=0))
    plt.legend(["ll_supp", "ll_alpha", "ll_supp_ips", "ll_alpha_ips"])
    plt.show()
    return LL
