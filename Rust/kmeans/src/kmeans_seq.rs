use crate::point::{euclidean_distance, mean, Point};
use rand::prelude::*;
use rand::seq::SliceRandom;

/// Returns a vector of centroids for each iteration (for state tracking), and final assignments
pub fn kmeans_seq(
    points: &[Point],
    k: usize,
    max_iters: usize,
    tolerance: f64,
    initial_centroids: Option<Vec<Point>>,
) -> (Vec<Vec<Point>>, Vec<usize>) {
    let mut rng = thread_rng();
    let mut centroids: Vec<Point> = match initial_centroids {
        Some(centroids) => centroids,
        None => points.choose_multiple(&mut rng, k).cloned().collect(),
    };
    let mut assignments = vec![0; points.len()];
    let mut states = vec![centroids.clone()];

    for _ in 0..max_iters {
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

        let mut max_shift = 0.0;
        for j in 0..k {
            let cluster_points: Vec<Point> = points
                .iter()
                .enumerate()
                .filter(|(i, _)| assignments[*i] == j)
                .map(|(_, p)| p.clone())
                .collect();
            if !cluster_points.is_empty() {
                let new_centroid = mean(&cluster_points);
                let shift = euclidean_distance(&centroids[j], &new_centroid);
                if shift > max_shift {
                    max_shift = shift;
                }
                centroids[j] = new_centroid;
            }
        }
        states.push(centroids.clone());
        if max_shift < tolerance {
            break;
        }
    }
    (states, assignments)
}
