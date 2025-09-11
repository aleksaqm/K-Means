import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import psutil
from sklearn.datasets import make_blobs

from kmeans import KMeans

def create_test_data(n_samples=1000, n_features=2, centers=5, cluster_std=1.5, random_state=42):
    x, y_true = make_blobs(n_samples=n_samples, centers=centers,
                          n_features=n_features, cluster_std=cluster_std,
                          random_state=random_state)
    return x, y_true

def run_experiment(n_samples, n_features, n_clusters, mode, n_jobs=None):
    times = []
    for _ in range(2):
        x, _ = create_test_data(n_samples=n_samples, n_features=n_features, centers=4)
        kmeans = KMeans(k=n_clusters, random_state=42)
        start = time.time()
        if mode == 'sequential':
            kmeans.fit(x)
        elif mode == 'parallel':
            kmeans.parallel_fit(x, n_jobs=n_jobs)
        else:
            raise ValueError('Unknown mode')
        times.append(time.time() - start)
    return times

def strong_scaling_experiment(result_file_path):
    n_samples = 100000
    n_features = 2
    n_clusters = 8
    max_cores = psutil.cpu_count(logical=True)
    results = []
    for cores in range(1, max_cores+1):
        seq_times = run_experiment(n_samples=n_samples, n_features=n_features, n_clusters=n_clusters, mode='sequential')
        mean_seq = np.mean(seq_times)
        std_seq = np.std(seq_times)

        par_times = run_experiment(n_samples=n_samples, n_features=n_features, n_clusters=n_clusters, mode='parallel', n_jobs=cores)
        mean_par = np.mean(par_times)
        std_par = np.std(par_times)
        speedup = mean_seq / mean_par if mean_par > 0 else 0
        efficiency = speedup / cores if cores > 0 else 0
        results.append({
            'Threads': cores,
            'MeanSeq': round(mean_seq, 4),
            'StdSeq': round(std_seq, 4),
            'MeanPar': round(mean_par, 4),
            'StdPar': round(std_par, 4),
            'Speedup': round(speedup, 2),
            'Efficiency': round(efficiency, 2)
        })
        print(f"ITERATION {cores}")

    with open(result_file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Threads','MeanSeq','StdSeq','MeanPar','StdPar','Speedup','Efficiency'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    return results

def weak_scaling_experiment(result_file_path):
    n_features = 2
    n_clusters = 8
    base_samples = 10000
    max_cores = psutil.cpu_count(logical=True)
    results = []
    for cores in range(1, max_cores+1):
        n_samples = base_samples * cores  # posao po jezgru konstantan

        # Seq
        seq_times = run_experiment(n_samples=n_samples, n_features=n_features, n_clusters=n_clusters, mode='sequential')
        mean_seq = np.mean(seq_times)
        std_seq = np.std(seq_times)
        # Par
        par_times = run_experiment(n_samples=n_samples, n_features=n_features, n_clusters=n_clusters, mode='parallel', n_jobs=cores)
        mean_par = np.mean(par_times)
        std_par = np.std(par_times)
        speedup = mean_seq / mean_par if mean_par > 0 else 0
        efficiency = speedup / cores if cores > 0 else 0
        results.append({
            'Threads': cores,
            'MeanSeq': round(mean_seq, 4),
            'StdSeq': round(std_seq, 4),
            'MeanPar': round(mean_par, 4),
            'StdPar': round(std_par, 4),
            'Speedup': round(speedup, 2),
            'Efficiency': round(efficiency, 2)
        })
        print(f"ITERATION {cores}")

    with open(result_file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Threads','MeanSeq','StdSeq','MeanPar','StdPar','Speedup','Efficiency'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    return results

def plot_scaling(results, law='amdahl', filename='scaling.png'):
    cores = [r['Threads'] for r in results]
    speedup = [r['Speedup'] for r in results]
    plt.figure(figsize=(8,6))
    plt.plot(cores, speedup, marker='o', label='Measured speedup')
    plt.plot(cores, cores, linestyle='--', color='gray', label='Ideal scaling')
    if law == 'amdahl':
        p = 0.90  # Example, can be tuned
        amdahl = [1/( (1-p) + p/c ) for c in cores]
        plt.plot(cores, amdahl, linestyle=':', color='red', label="Amdahl's law (p=0.90)")
    elif law == 'gustafson':
        p = 0.90
        gustafson = [c - (1-p)*(c-1) for c in cores]
        plt.plot(cores, gustafson, linestyle=':', color='red', label="Gustafson's law (p=0.90)")
    plt.xlabel('Number of cores')
    plt.ylabel('Speedup')
    plt.title('Scaling experiment')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    print('\nRunning strong scaling experiment')
    result_file_path = "experiments_files/strong_scaling_python5.csv"
    strong_results = strong_scaling_experiment(result_file_path)
    plot_scaling(strong_results, law='amdahl', filename='experiments_files/strong_scaling_python5.png')
    print('Strong scaling results saved.')

    print('\nRunning weak scaling experiment')
    result_file_path = "experiments_files/weak_scaling_python5.csv"
    weak_results = weak_scaling_experiment(result_file_path)
    plot_scaling(weak_results, law='gustafson', filename='experiments_files/weak_scaling_python5.png')
    print('Weak scaling results saved.')
    print('\nDone.')