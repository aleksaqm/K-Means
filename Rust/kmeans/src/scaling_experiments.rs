mod kmeans_par;
mod kmeans_seq;
mod point;

use kmeans_par::*;
use kmeans_seq::*;
use plotters::prelude::*;
use point::Point;
use rand::prelude::*;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn create_test_data(n_points: usize) -> Vec<Point> {
    let mut rng = thread_rng();
    (0..n_points)
        .map(|_| Point {
            x: rng.gen_range(0.0..100.0),
            y: rng.gen_range(0.0..100.0),
        })
        .collect()
}

fn mean_std(times: &[f64]) -> (f64, f64) {
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let std = (times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64).sqrt();
    (mean, std)
}

fn strong_scaling(
    n_points: usize,
    k: usize,
    max_iters: usize,
    tolerance: f64,
    max_threads: usize,
    n_runs: usize,
) -> Vec<(usize, f64, f64, f64, f64, f64, f64)> {
    let mut results = Vec::new();
    for threads in 1..=max_threads {
        let mut seq_times = Vec::new();
        let mut par_times = Vec::new();
        for _ in 0..n_runs {
            let points = create_test_data(n_points);
            let initial_centroids: Vec<Point> = points
                .choose_multiple(&mut thread_rng(), k)
                .cloned()
                .collect();
            let start_seq = Instant::now();
            let _ = kmeans_seq::kmeans_seq(
                &points,
                k,
                max_iters,
                tolerance,
                Some(initial_centroids.clone()),
            );
            let seq_time = start_seq.elapsed().as_secs_f64();
            seq_times.push(seq_time);

            let start_par = Instant::now();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let _ = kmeans_par::kmeans_par(
                    &points,
                    k,
                    max_iters,
                    tolerance,
                    Some(initial_centroids.clone()),
                );
            });
            let par_time = start_par.elapsed().as_secs_f64();
            par_times.push(par_time);
        }
        let (mean_seq, std_seq) = mean_std(&seq_times);
        let (mean_par, std_par) = mean_std(&par_times);
        let speedup = mean_seq / mean_par;
        let efficiency = speedup / threads as f64;
        results.push((
            threads, mean_seq, std_seq, mean_par, std_par, speedup, efficiency,
        ));
    }
    results
}

fn weak_scaling(
    base_points: usize,
    k: usize,
    max_iters: usize,
    tolerance: f64,
    max_threads: usize,
    n_runs: usize,
) -> Vec<(usize, f64, f64, f64, f64, f64, f64)> {
    let mut results = Vec::new();
    for threads in 1..=max_threads {
        let n_points = base_points * threads;
        let mut seq_times = Vec::new();
        let mut par_times = Vec::new();
        for _ in 0..n_runs {
            let points = create_test_data(n_points);
            let initial_centroids: Vec<Point> = points
                .choose_multiple(&mut thread_rng(), k)
                .cloned()
                .collect();
            let start_seq = Instant::now();
            let _ = kmeans_seq::kmeans_seq(
                &points,
                k,
                max_iters,
                tolerance,
                Some(initial_centroids.clone()),
            );
            let seq_time = start_seq.elapsed().as_secs_f64();
            seq_times.push(seq_time);

            let start_par = Instant::now();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let _ = kmeans_par::kmeans_par(
                    &points,
                    k,
                    max_iters,
                    tolerance,
                    Some(initial_centroids.clone()),
                );
            });
            let par_time = start_par.elapsed().as_secs_f64();
            par_times.push(par_time);
        }
        let (mean_seq, std_seq) = mean_std(&seq_times);
        let (mean_par, std_par) = mean_std(&par_times);
        let speedup = mean_seq / mean_par;
        let efficiency = speedup / threads as f64;
        results.push((
            threads, mean_seq, std_seq, mean_par, std_par, speedup, efficiency,
        ));
    }
    results
}

fn save_csv(filename: &str, results: &[(usize, f64, f64, f64, f64, f64, f64)]) {
    let mut file = File::create(filename).unwrap();
    writeln!(
        file,
        "Threads,MeanSeq,StdSeq,MeanPar,StdPar,Speedup,Efficiency"
    )
    .unwrap();
    for r in results {
        writeln!(
            file,
            "{},{:.4},{:.4},{:.4},{:.4},{:.2},{:.2}",
            r.0, r.1, r.2, r.3, r.4, r.5, r.6
        )
        .unwrap();
    }
}

fn plot_scaling(
    filename: &str,
    results: &[(usize, f64, f64, f64, f64, f64, f64)],
    law: &str,
    p: f64,
) {
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let cores: Vec<usize> = results.iter().map(|r| r.0).collect();
    let speedup: Vec<f64> = results.iter().map(|r| r.5).collect();
    let max_cores = *cores.last().unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("Scaling experiment", ("sans-serif", 30))
        .margin(40)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(1..max_cores, 0.0..(max_cores as f64 + 1.0))
        .unwrap();
    chart
        .configure_mesh()
        .x_desc("Number of cores")
        .y_desc("Speedup")
        .draw()
        .unwrap();
    chart
        .draw_series(LineSeries::new(
            cores.iter().zip(speedup.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))
        .unwrap()
        .label("Measured speedup")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 1, y)], &BLUE));
    chart
        .draw_series(LineSeries::new(
            cores.iter().map(|&x| (x, x as f64)),
            &BLACK,
        ))
        .unwrap()
        .label("Ideal scaling")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 1, y)], &BLACK));
    if law == "amdahl" {
        chart
            .draw_series(LineSeries::new(
                cores.iter().map(|&c| (c, 1.0 / ((1.0 - p) + p / c as f64))),
                &RED,
            ))
            .unwrap()
            .label(&format!("Amdahl's law p={}", p))
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 1, y)], &RED));
    } else if law == "gustafson" {
        chart
            .draw_series(LineSeries::new(
                cores
                    .iter()
                    .map(|&c| (c, c as f64 - (1.0 - p) * (c as f64 - 1.0))),
                &RED,
            ))
            .unwrap()
            .label(&format!("Gustafson's law p={}", p))
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 1, y)], &RED));
    }
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
}

fn main() {
    let k = 4;
    let max_iters = 100;
    let tolerance = 1e-3;
    let max_threads = 8;
    let n_runs = 2;
    let n_points = 100000;
    let base_points = 200000;
    let p = 0.9; // Parallel fraction

    println!("Running strong scaling experiment...");
    let strong_results = strong_scaling(n_points, k, max_iters, tolerance, max_threads, n_runs);
    save_csv("strong_scaling_rust13.csv", &strong_results);
    plot_scaling("strong_scaling_rust13.png", &strong_results, "amdahl", p);
    println!("Strong scaling done.");

    // println!("Running weak scaling experiment...");
    // let weak_results = weak_scaling(base_points, k, max_iters, tolerance, max_threads, n_runs);
    // save_csv("weak_scaling_rust6.csv", &weak_results);
    // plot_scaling("weak_scaling_rust6.png", &weak_results, "gustafson", p);
    // println!("Weak scaling done.");
}
