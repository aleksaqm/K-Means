use crate::point::{euclidean_distance, Point};
use rand::prelude::*;
use rand::seq::SliceRandom;

pub fn kmeans_seq(
    points: &[Point],
    k: usize,
    max_iters: usize,
    tolerance: f64,
    initial_centroids: Option<Vec<Point>>,
) -> (Vec<Point>, Vec<usize>) {
    let mut rng = thread_rng();
    let mut centroids: Vec<Point> = match initial_centroids {
        Some(centroids) => centroids,
        None => points.choose_multiple(&mut rng, k).cloned().collect(),
    };
    let mut assignments = vec![0; points.len()];

    for i in 0..max_iters {
        for (i, point) in points.iter().enumerate() {
            let mut min_dist = f64::MAX;
            let mut cluster = 0;
            for (j, centroid) in centroids.iter().enumerate() {
                let dist = euclidean_distance(point, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    cluster = j;
                }
            }
            assignments[i] = cluster;
        }

        let mut sums = vec![Point::zero(); k];
        let mut counts = vec![0usize; k];
        for (point, &cluster) in points.iter().zip(assignments.iter()) {
            sums[cluster] = sums[cluster].add(point);
            counts[cluster] += 1;
        }

        let mut max_shift = 0.0;
        for j in 0..k {
            if counts[j] > 0 {
                let new_centroid = sums[j].div(counts[j] as f64);
                let shift = euclidean_distance(&centroids[j], &new_centroid);
                if shift > max_shift {
                    max_shift = shift;
                }
                centroids[j] = new_centroid;
            }
        }
        println!("Iteration {i}, shift = {max_shift}");
        if max_shift < tolerance {
            break;
        }
    }
    (centroids, assignments)
}
