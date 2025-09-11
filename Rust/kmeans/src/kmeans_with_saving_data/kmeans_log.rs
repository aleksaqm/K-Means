use crate::point::{euclidean_distance, Point};
use rand::prelude::*;
use serde_json::json;
use std::fs::File;
use std::io::Write;

pub fn kmeans_seq_log(
    points: &[Point],
    k: usize,
    max_iters: usize,
    tolerance: f64,
    initial_centroids: Option<Vec<Point>>,
    json_path: &str,
) -> (Vec<Point>, Vec<usize>) {
    let mut rng = thread_rng();
    let mut centroids: Vec<Point> = match initial_centroids {
        Some(centroids) => centroids,
        None => points.choose_multiple(&mut rng, k).cloned().collect(),
    };
    let mut assignments = vec![0; points.len()];

    let mut json_obj = json!({
        "points": points.iter().map(|p| vec![p.x, p.y]).collect::<Vec<_>>(),
        "iterations": []
    });

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
        let iter_obj = json!({
            "centroids": centroids.iter().map(|c| vec![c.x, c.y]).collect::<Vec<_>>(),
            "labels": assignments.clone()
        });
        json_obj["iterations"]
            .as_array_mut()
            .unwrap()
            .push(iter_obj);

        if max_shift < tolerance {
            break;
        }
    }
    let mut file = File::create(json_path).expect("Cannot create JSON file");
    let json_str = serde_json::to_string_pretty(&json_obj).unwrap();
    file.write_all(json_str.as_bytes())
        .expect("Cannot write JSON");
    (centroids, assignments)
}
