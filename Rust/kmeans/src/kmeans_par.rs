use crate::point::{euclidean_distance, Point};
use rand::prelude::*;
use rand::seq::SliceRandom;
use rayon::prelude::*;

pub fn kmeans_par(
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
        assignments
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, assign)| {
                let point = &points[i];
                let cluster = centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        euclidean_distance(point, a)
                            .partial_cmp(&euclidean_distance(point, b))
                            .unwrap()
                    })
                    .map(|(j, _)| j)
                    .unwrap_or(0);
                *assign = cluster;
            });

        // Parallel reduction for sums and counts
        let (sums, counts) = points
            .par_iter()
            .zip(assignments.par_iter())
            .fold(
                || (vec![Point::zero(); k], vec![0usize; k]),
                |mut acc, (point, &cluster)| {
                    acc.0[cluster] = acc.0[cluster].add(point);
                    acc.1[cluster] += 1;
                    acc
                },
            )
            .reduce(
                || (vec![Point::zero(); k], vec![0usize; k]),
                |(mut sums1, mut counts1), (sums2, counts2)| {
                    for j in 0..k {
                        sums1[j] = sums1[j].add(&sums2[j]);
                        counts1[j] += counts2[j];
                    }
                    (sums1, counts1)
                },
            );

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
