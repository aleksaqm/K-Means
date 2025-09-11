mod kmeans_par;
mod kmeans_seq;
mod point;
use crate::point::Point;
use rand::prelude::*;
use std::time::Instant;

fn main() {
    let n_points = 2;
    let k = 4;
    let max_iters = 100;
    let tolerance = 0.001;
    let mut rng = thread_rng();
    let points: Vec<Point> = (0..n_points)
        .map(|_| Point {
            x: rng.gen_range(0.0..100.0),
            y: rng.gen_range(0.0..100.0),
        })
        .collect();

    let initial_centroids: Vec<Point> = points.choose_multiple(&mut rng, k).cloned().collect();

    let start_seq = Instant::now();
    let (seq_centroids, seq_assignments) = kmeans_seq::kmeans_seq(
        &points,
        k,
        max_iters,
        tolerance,
        Some(initial_centroids.clone()),
    );
    let duration_seq = start_seq.elapsed();
    println!("Sequential K-Means: Final centroids:");
    for (i, c) in seq_centroids.iter().enumerate() {
        println!("Cluster {}: ({:.2}, {:.2})", i, c.x, c.y);
    }
    println!("Time elapsed: {:.2?}", duration_seq);

    let start_par = Instant::now();
    let (par_centroids, par_assignments) = kmeans_par::kmeans_par(
        &points,
        k,
        max_iters,
        tolerance,
        Some(initial_centroids.clone()),
    );
    let duration_par = start_par.elapsed();
    println!("\nParallel K-Means: Final centroids:");
    for (i, c) in par_centroids.iter().enumerate() {
        println!("Cluster {}: ({:.2}, {:.2})", i, c.x, c.y);
    }
    println!("Time elapsed: {:.2?}", duration_par);

    println!("First Seq assignment {}", seq_assignments[0]);
    println!("First Par assignment {}", par_assignments[0]);
}
