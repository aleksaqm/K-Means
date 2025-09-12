mod kmeans_log;
mod point;
use crate::point::Point;
use kmeans_log::kmeans_seq_log;
use rand::prelude::*;
use std::{fs, path::Path};

fn create_test_data(n_points: usize) -> Vec<Point> {
    let mut rng = thread_rng();
    (0..n_points)
        .map(|_| Point {
            x: rng.gen_range(0.0..100.0),
            y: rng.gen_range(0.0..100.0),
        })
        .collect()
}

fn next_log_filename(log_dir: &str) -> String {
    let mut idx = 1;
    let mut filename = format!("{}/kmeans_log.json", log_dir);
    while Path::new(&filename).exists() {
        idx += 1;
        filename = format!("{}/kmeans_log{}.json", log_dir, idx);
    }
    filename
}

fn main() {
    let log_dir = "log_files";
    fs::create_dir_all(log_dir).expect("Cannot create log_files directory");

    let n_points = 1000;
    let k = 3;
    let max_iters = 20;
    let tolerance = 0.1;
    let points = create_test_data(n_points);
    let initial_centroids: Vec<Point> = points
        .choose_multiple(&mut thread_rng(), k)
        .cloned()
        .collect();

    let json_path = next_log_filename(log_dir);
    println!("Running sequential KMeans with logging...");
    kmeans_seq_log(
        &points,
        k,
        max_iters,
        tolerance,
        Some(initial_centroids),
        &json_path,
    );
    println!("Log file created at: {}", json_path);
}
