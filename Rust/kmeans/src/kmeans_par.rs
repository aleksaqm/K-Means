use crate::point::{euclidean_distance, mean, Point};
use rand::prelude::*;
use rand::seq::SliceRandom;
use rayon::prelude::*;

/// Returns a vector of centroids for each iteration (for state tracking), and final assignments
pub fn kmeans_par(
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
        assignments = points
            .par_iter()
            .map(|point| {
                centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        euclidean_distance(point, a)
                            .partial_cmp(&euclidean_distance(point, b))
                            .unwrap()
                    })
                    .map(|(j, _)| j)
                    .unwrap_or(0)
            })
            .collect();

        let mut max_shift = 0.0;
        let new_centroids: Vec<Point> = (0..k)
            .into_par_iter()
            .map(|j| {
                let cluster_points: Vec<Point> = points
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| assignments[*i] == j)
                    .map(|(_, p)| p.clone())
                    .collect();
                if !cluster_points.is_empty() {
                    mean(&cluster_points)
                } else {
                    centroids[j].clone()
                }
            })
            .collect();

        for (j, new_centroid) in new_centroids.iter().enumerate() {
            let shift = euclidean_distance(&centroids[j], new_centroid);
            if shift > max_shift {
                max_shift = shift;
            }
            centroids[j] = new_centroid.clone();
        }
        states.push(centroids.clone());
        if max_shift < tolerance {
            break;
        }
    }
    (states, assignments)
}
