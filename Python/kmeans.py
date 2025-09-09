import numpy as np
from sklearn.datasets import make_blobs

import time
import warnings
from multiprocessing import Pool, cpu_count
from multiprocessing import shared_memory

warnings.filterwarnings('ignore')


class KMeans:
    def __init__(self, k=3, max_iters=100, tolerance=1e-4, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self._shared_memory = None

    def _initialize_centroids(self, points):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = points.shape
        centroids = np.zeros((self.k, n_features))
        
        for i in range(self.k):
            centroids[i] = points[np.random.choice(n_samples)]
        
        return centroids

    @staticmethod
    def _calculate_distance(points, centroids):
        distances = []
        for x in points:
            point_distances = []
            for centroid in centroids:
                dist = 0.0
                for a, b in zip(x, centroid):
                    dist += (a - b) ** 2
                point_distances.append(dist ** 0.5)
            distances.append(point_distances)
        return distances

    @staticmethod
    def _assign_clusters(distances):
        labels = []
        for dist in distances:
            min_idx = 0
            min_val = dist[0]
            for idx, val in enumerate(dist):
                if val < min_val:
                    min_val = val
                    min_idx = idx
            labels.append(min_idx)
        return np.array(labels)

    def _update_centroids(self, x):
        centroids = []
        for i in range(self.k):
            sum_vec = [0.0] * x.shape[1]
            count = 0
            for idx, label in enumerate(self.labels):
                if label == i:
                    for j in range(x.shape[1]):
                        sum_vec[j] += x[idx][j]
                    count += 1
            if count > 0:
                centroids.append([val / count for val in sum_vec])
            else:
                centroids.append(list(x[np.random.choice(x.shape[0])]))
        self.centroids = np.array(centroids)
    
    def fit(self, points, verbose=False):
        self.centroids = self._initialize_centroids(points)
        for iteration in range(self.max_iters):
            old_centroids = self.centroids.copy()
            
            # Calculate distances and assign clusters
            distances = self._calculate_distance(points, self.centroids)
            self.labels = self._assign_clusters(distances)

            # Update centroids
            self._update_centroids(points)
            
            # Check for convergence
            centroid_shift = np.linalg.norm(self.centroids - old_centroids)
            
            if verbose:
                print(f"Iteration {iteration + 1}: Centroid shift = {centroid_shift:.6f}")
            
            if centroid_shift < self.tolerance:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        
        return self

    @staticmethod
    def _assign_clusters_chunk_shared(args):
        shm_name, shape, dtype, start_idx, end_idx, centroids = args

        existing_shm = shared_memory.SharedMemory(name=shm_name)
        points = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

        points_chunk = points[start_idx:end_idx]

        distances = KMeans._calculate_distance(points_chunk, centroids)
        labels = KMeans._assign_clusters(distances)

        existing_shm.close()
        return labels

    def _assign_clusters_parallel(self, points, pool):
        n_cores = cpu_count()
        chunk_size = max(1, len(points) // (n_cores * 2))

        chunks = []
        for i in range(0, len(points), chunk_size):
            start_idx = i
            end_idx = min(i + chunk_size, len(points))
            chunks.append((
                self._shared_memory['shm'].name,
                self._shared_memory['shape'],
                self._shared_memory['dtype'],
                start_idx,
                end_idx,
                self.centroids.copy()
            ))

        results = pool.map(KMeans._assign_clusters_chunk_shared, chunks)

        all_labels = []
        for chunk_labels in results:
            all_labels.extend(chunk_labels)

        self.labels = np.array(all_labels)

    @staticmethod
    def _partial_sum_chunk_shared(args):
        shm_name, shape, dtype, idx_range, labels_chunk, k = args

        existing_shm = shared_memory.SharedMemory(name=shm_name)
        points = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        start, end = idx_range
        sums = np.zeros((k, points.shape[1]))
        counts = np.zeros(k, dtype=int)
        for i, x in enumerate(points[start:end]):
            label = labels_chunk[i]
            sums[label] += x
            counts[label] += 1
        existing_shm.close()
        return sums, counts

    def _update_centroids_parallel(self, points, pool):
        n_cores = cpu_count()
        chunk_size = int(np.ceil(points.shape[0] / (n_cores * 2)))

        # Use existing shared memory instead of creating new one
        chunks = []
        for i in range(0, points.shape[0], chunk_size):
            start = i
            end = min(i + chunk_size, points.shape[0])
            chunks.append((
                self._shared_memory['shm'].name,  # Use existing shared memory
                self._shared_memory['shape'],
                self._shared_memory['dtype'],
                (start, end),
                self.labels[start:end],
                self.k
            ))

        results = pool.map(KMeans._partial_sum_chunk_shared, chunks)

        total_sums = np.sum([r[0] for r in results], axis=0)
        total_counts = np.sum([r[1] for r in results], axis=0)

        new_centroids = np.zeros((self.k, points.shape[1]))
        for i in range(self.k):
            if total_counts[i] > 0:
                new_centroids[i] = total_sums[i] / total_counts[i]
            else:
                new_centroids[i] = points[np.random.choice(points.shape[0])]

        self.centroids = new_centroids

    def _create_shared_memory(self, points):
        if self._shared_memory is not None:
            return
        self._shared_memory = {
            'shm': shared_memory.SharedMemory(create=True, size=points.nbytes),
            'shape': points.shape,
            'dtype': points.dtype
        }
        shm_points = np.ndarray(points.shape, dtype=points.dtype,
                                buffer=self._shared_memory['shm'].buf)
        np.copyto(shm_points, points)

    def _cleanup_shared_memory(self):
        if self._shared_memory is not None:
            self._shared_memory['shm'].close()
            self._shared_memory['shm'].unlink()
            self._shared_memory = None


    def parallel_fit(self, points, verbose=False):
        self.centroids = self._initialize_centroids(points)
        self._create_shared_memory(points)

        n_cores = cpu_count()
        try:
            with Pool(n_cores) as pool:  # <- Pool se kreira samo jednom
                for iteration in range(self.max_iters):
                    old_centroids = self.centroids.copy()

                    self._assign_clusters_parallel(points, pool)
                    self._update_centroids_parallel(points, pool)

                    centroid_shift = np.linalg.norm(self.centroids - old_centroids)
                    if verbose:
                        print(f"[PARALLEL] Iteration {iteration + 1}, shift={centroid_shift:.6f}")
                    if centroid_shift < self.tolerance:
                        if verbose:
                            print(f"[PARALLEL] Converged after {iteration + 1} iterations")
                        break
        finally:
            self._cleanup_shared_memory()
        return self



def create_test_data(n_samples=1000, n_features=2, centers=5, cluster_std=1.5, random_state=42):
    x, y_true = make_blobs(n_samples=n_samples, centers=centers,
                          n_features=n_features, cluster_std=cluster_std,
                          random_state=random_state)
    return x, y_true


def main():
    test_data, y_true = create_test_data(n_samples=50000, centers=8, random_state=42)
    print(f"Created dataset with {test_data.shape[0]} samples and {test_data.shape[1]} features")

    # Run sequential K-means
    print("\n1. Running sequential K-means...")
    kmeans = KMeans(k=8, random_state=42)
    start_time = time.time()
    kmeans.fit(test_data, verbose=True)
    seq_time = time.time() - start_time
    print(f"Sequential K-means completed in {seq_time:.4f} seconds")

    # Run parallel K-means
    print("\n2. Running parallel K-means...")
    start_time = time.time()
    kmeans.parallel_fit(test_data, verbose=True)
    par_time = time.time() - start_time
    print(f"Parallel K-means completed in {par_time:.4f} seconds")

if __name__ == "__main__":
    main()