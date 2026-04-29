import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

class SpeakerClustering:
    def __init__(self, threshold=0.75):
        """
        threshold: The distance threshold for merging clusters. 
                   Lower = more clusters (speakers), Higher = fewer clusters.
        """
        self.threshold = threshold

    def cluster(self, embeddings):
        """
        embeddings: List or array of speaker embeddings.
        Returns: Cluster labels for each embedding.
        """
        if len(embeddings) < 2:
            return np.zeros(len(embeddings), dtype=int)

        # Compute cosine distance matrix (1 - cosine similarity)
        dist_matrix = cosine_distances(embeddings)

        # Agglomerative clustering with precomputed distance
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=self.threshold
        )

        labels = clustering.fit_predict(dist_matrix)
        return labels
