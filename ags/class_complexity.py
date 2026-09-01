"""
CAGS: Class-Adaptive Guidance Strength

Computes per-class complexity scores to determine the optimal guidance strength
for each class during diffusion sampling.

Key insight: Classes with higher intra-class complexity (more modes, higher entropy)
need stronger guidance to ensure all sub-modes are covered. Simple classes can rely
more on the diffusion prior with weaker guidance.

Complexity(c) = alpha * mode_count(c) / max_k + beta * intra_class_entropy(c)

guidance_strength(c) = sigmoid(Complexity(c))
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import defaultdict
from tqdm import tqdm
import time
import pickle
import os


class ClassComplexityAnalyzer:
    """
    Analyzes per-class complexity to determine adaptive guidance strength.

    The complexity score combines:
      1. Mode count: Number of distinct clusters (sub-modes) in feature space
      2. Intra-class entropy: Distribution uniformity across clusters

    A higher complexity score leads to stronger guidance during diffusion sampling.
    """

    def __init__(
        self,
        feature_extractor=None,
        n_clusters_range=(2, 20),
        alpha=0.5,
        beta=0.5,
        use_pca=True,
        pca_components=4,
        use_silhouette=True,
        max_k_method="silhouette",
    ):
        """
        Args:
            feature_extractor: Model to extract features (e.g., VAE encoder)
            n_clusters_range: Range of K for K-means clustering
            alpha: Weight for mode count in complexity score
            beta: Weight for entropy in complexity score
            use_pca: Whether to apply PCA before clustering
            pca_components: Number of PCA components if used
            use_silhouette: Whether to use silhouette score for optimal K selection
            max_k_method: Method for selecting optimal K ('silhouette' or 'elbow')
        """
        self.feature_extractor = feature_extractor
        self.n_clusters_range = n_clusters_range
        self.alpha = alpha
        self.beta = beta
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.use_silhouette = use_silhouette
        self.max_k_method = max_k_method
        self.max_k = n_clusters_range[1]

        self.complexity_scores = {}
        self.mode_counts = {}
        self.cluster_centers = {}
        self.cluster_centers_path = {}
        self.mode_id_per_class = {}

    def _find_optimal_k(self, X, k_min=2, k_max=20):
        """Find optimal number of clusters using silhouette score or elbow method."""
        k_max = min(k_max, len(X) - 1)
        if k_max < k_min:
            return k_min, np.zeros(len(X))

        if self.use_pca and X.shape[1] > self.pca_components:
            pca = PCA(n_components=self.pca_components)
            X_pca = pca.fit_transform(X)
        else:
            X_pca = X

        if self.max_k_method == "silhouette" and self.use_silhouette:
            best_k = k_min
            best_score = -1
            for k in range(k_min, k_max + 1):
                kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
                labels = kmeans.fit_predict(X_pca)
                if len(set(labels)) > 1:
                    score = silhouette_score(X_pca, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            return best_k, KMeans(n_clusters=best_k, random_state=0, n_init=10).fit_predict(X_pca)
        else:
            # Elbow method using inertia
            inertias = []
            for k in range(k_min, k_max + 1):
                kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
                kmeans.fit(X_pca)
                inertias.append(kmeans.inertia_)

            # Find elbow point
            diffs = np.diff(inertias)
            if len(diffs) > 1:
                second_diffs = np.diff(diffs)
                elbow_idx = np.argmax(np.abs(second_diffs)) + 1
                best_k = k_min + elbow_idx
            else:
                best_k = k_min

            best_k = max(k_min, min(best_k, k_max))
            kmeans = KMeans(n_clusters=best_k, random_state=0, n_init=10)
            return best_k, kmeans.fit_predict(X_pca)

    def compute_complexity(self, features, class_id):
        """
        Compute complexity score for a single class.

        Args:
            features: numpy array of shape (N, D) - features for this class
            class_id: class identifier

        Returns:
            complexity_score: float in [0, 1]
            n_modes: int - number of detected modes
            cluster_centers: numpy array of shape (n_modes, D)
        """
        X = np.stack(features) if isinstance(features, list) else features

        # Find optimal K and cluster
        optimal_k, labels = self._find_optimal_k(
            X, k_min=self.n_clusters_range[0], k_max=self.n_clusters_range[1]
        )

        # Get cluster centers (closest real points to centroids)
        kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init=10)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_

        # Find closest real points to centroids
        closest_points = []
        for center in centers:
            closest_idx = np.argmin(np.sum((X - center) ** 2, axis=1))
            closest_points.append(X[closest_idx])
        cluster_centers = np.stack(closest_points)

        # Compute intra-class entropy
        proportions = np.bincount(labels, minlength=optimal_k) / len(labels)
        proportions = proportions[proportions > 0]
        entropy = -np.sum(proportions * np.log(proportions + 1e-8))
        normalized_entropy = entropy / np.log(optimal_k) if optimal_k > 1 else 0.0

        # Complexity score
        mode_count_score = optimal_k / self.max_k
        complexity_score = self.alpha * mode_count_score + self.beta * normalized_entropy

        # Sigmoid normalization to [0, 1]
        complexity_score = 1.0 / (1.0 + np.exp(-5 * (complexity_score - 0.5)))

        self.complexity_scores[class_id] = float(complexity_score)
        self.mode_counts[class_id] = optimal_k
        self.cluster_centers[class_id] = cluster_centers
        self.mode_id_per_class[class_id] = labels

        return complexity_score, optimal_k, cluster_centers

    def analyze_all_classes(self, features_per_class, paths_per_class=None):
        """
        Compute complexity for all classes.

        Args:
            features_per_class: dict {class_id: list of feature arrays}
            paths_per_class: dict {class_id: list of image paths} (optional)

        Returns:
            complexity_scores: dict {class_id: float}
            mode_counts: dict {class_id: int}
            cluster_centers: dict {class_id: numpy array}
        """
        self.features_per_class = features_per_class
        for class_id in tqdm(features_per_class.keys(), desc="Computing class complexity"):
            start_time = time.time()
            features = features_per_class[class_id]
            complexity, n_modes, centers = self.compute_complexity(features, class_id)

            if paths_per_class is not None and class_id in paths_per_class:
                # Store paths for closest points
                self.cluster_centers_path[class_id] = [
                    paths_per_class[class_id][
                        np.argmin(np.sum(
                            np.stack(features) - centers[i], axis=1
                        ) ** 2)
                    ]
                    for i in range(n_modes)
                ]

            end_time = time.time()
            print(
                f"Class {class_id}: complexity={complexity:.4f}, "
                f"modes={n_modes}, time={end_time - start_time:.2f}s"
            )

        return self.complexity_scores, self.mode_counts, self.cluster_centers

    def get_guidance_strength(self, class_id, scale_range=(0.05, 0.5)):
        """
        Map complexity score to guidance strength.

        Args:
            class_id: class identifier
            scale_range: (min_scale, max_scale) for guidance strength

        Returns:
            guidance_strength: float in [min_scale, max_scale]
        """
        if class_id not in self.complexity_scores:
            return (scale_range[0] + scale_range[1]) / 2

        complexity = self.complexity_scores[class_id]
        min_s, max_s = scale_range
        return min_s + complexity * (max_s - min_s)

    def save(self, path):
        """Save analysis results to file."""
        data = {
            "complexity_scores": self.complexity_scores,
            "mode_counts": self.mode_counts,
            "cluster_centers": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.cluster_centers.items()
            },
            "cluster_centers_path": self.cluster_centers_path,
            "mode_id_per_class": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.mode_id_per_class.items()
            },
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        """Load analysis results from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.complexity_scores = data["complexity_scores"]
        self.mode_counts = data["mode_counts"]
        self.cluster_centers = {
            k: np.array(v) for k, v in data["cluster_centers"].items()
        }
        self.cluster_centers_path = data.get("cluster_centers_path", {})
        self.mode_id_per_class = data.get("mode_id_per_class", {})

    def compute_clusters_for_ipc(self, ipc, use_pca=True, closest_point=True):
        """Compute cluster centers with K=IPC for generation.

        CAGS uses optimal K for complexity scoring, but generation needs
        exactly IPC mode centers to guide each of the IPC generated images.
        """
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        ipc_clusters = {}
        for class_label, features in self.features_per_class.items():
            X = np.stack(features)
            if len(X) < ipc:
                indices = np.random.choice(len(X), ipc, replace=True)
                ipc_clusters[class_label] = X[indices]
                continue

            if use_pca and X.shape[1] > 4:
                pca = PCA(n_components=4)
                X_pca = pca.fit_transform(X)
                kmeans = KMeans(n_clusters=ipc, random_state=0, n_init="auto").fit(X_pca)
            else:
                X_pca = X
                kmeans = KMeans(n_clusters=ipc, random_state=0, n_init="auto").fit(X)

            if closest_point:
                centers = kmeans.cluster_centers_
                closest = []
                for center in centers:
                    idx = np.argmin(np.sum((X_pca - center) ** 2, axis=1))
                    closest.append(X[idx])
                ipc_clusters[class_label] = np.stack(closest)
            else:
                ipc_clusters[class_label] = kmeans.cluster_centers_

        return ipc_clusters
