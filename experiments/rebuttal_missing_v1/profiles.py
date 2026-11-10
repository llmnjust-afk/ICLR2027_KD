"""Estimator interventions; the original mode_targets.npz is never changed."""
from pathlib import Path
import numpy as np

PROFILES = ['vae_k5', 'vae_k10', 'vae_pca16', 'vae_ward10', 'resnet18_auto']


def ward_profile(features, classes, baseline, seed=2027):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA
    from rebuttal.common import seed_for
    import copy
    result = copy.deepcopy(baseline)
    result.update(fixed_k=10, pca_components=4, clustering='Ward agglomerative, Euclidean')
    for cls in classes:
        np.random.seed(seed_for('complexity_class', seed, cls))
        x = np.load(Path(features) / (cls + '.npy'), mmap_mode='r')
        projected = PCA(n_components=4).fit_transform(x)
        labels = AgglomerativeClustering(n_clusters=10, linkage='ward').fit_predict(projected)
        counts = np.bincount(labels, minlength=10)
        p = counts[counts > 0] / len(x)
        result['per_class'][cls]['k'] = 10
        result['per_class'][cls]['entropy_normalized'] = float(-np.sum(p * np.log(p + 1e-8)) / np.log(10))
        print('Ward profile:', cls, flush=True)
    return result


def compute(name, prepared, work):
    from rebuttal.prepare import profile_from_features
    from rebuttal.common import read_json
    base = read_json(Path(prepared) / 'profiles/vae_auto.json')
    classes, seed = base['classes'], base['random_seed']
    features = Path(prepared) / 'features'
    if name == 'vae_ward10':
        return ward_profile(features, classes, base, seed)
    fixed = {'vae_k5': 5, 'vae_k10': 10}.get(name)
    dimension = 16 if name == 'vae_pca16' else base['pca_components']
    if name == 'resnet18_auto':
        features = Path(work) / 'features_resnet18'
    return profile_from_features(features, classes, base['k_max'], dimension, fixed, seed)
