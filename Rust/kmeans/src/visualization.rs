use plotters::prelude::*;
use serde_json::Value;
use std::fs::{create_dir_all, File};
use std::io::BufReader;
use std::path::Path;

fn next_vis_folder(base: &str) -> String {
    let mut idx = 1;
    loop {
        let folder = format!("{}/visualization{}", base, idx);
        if !Path::new(&folder).exists() {
            return folder;
        }
        idx += 1;
    }
}

fn main() {
    let file = File::open("log_files/kmeans_log4.json").expect("Cannot open JSON file");
    let reader = BufReader::new(file);
    let data: Value = serde_json::from_reader(reader).expect("Cannot parse JSON");

    let points = data["points"].as_array().unwrap();
    let iterations = data["iterations"].as_array().unwrap();

    // Boje za klastere
    let colors = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &YELLOW, &BLACK];

    let vis_base = "visualization_photos";
    let vis_folder = next_vis_folder(vis_base);
    create_dir_all(&vis_folder).expect("Cannot create visualization folder");

    for (iter_idx, iter) in iterations.iter().enumerate() {
        let centroids = iter["centroids"].as_array().unwrap();
        let labels = iter["labels"].as_array().unwrap();

        let filename = format!("{}/iter_{}.png", vis_folder, iter_idx);
        let root = BitMapBackend::new(&filename, (800, 800)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root)
            .margin(20)
            .caption(format!("KMeans Iteration {iter_idx}"), ("sans-serif", 30))
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0f64..100f64, 0f64..100f64)
            .unwrap();

        chart.configure_mesh().draw().unwrap();

        // Crtanje tacaka
        for (i, p) in points.iter().enumerate() {
            let x = p[0].as_f64().unwrap();
            let y = p[1].as_f64().unwrap();
            let label = labels[i].as_u64().unwrap() as usize;
            let color = colors[label % colors.len()];
            chart
                .draw_series(PointSeries::of_element(
                    vec![(x, y)],
                    2,
                    color,
                    &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
                ))
                .unwrap();
        }

        // Crtanje centroida
        for (j, c) in centroids.iter().enumerate() {
            let x = c[0].as_f64().unwrap();
            let y = c[1].as_f64().unwrap();
            let color = colors[j % colors.len()];
            chart
                .draw_series(PointSeries::of_element(
                    vec![(x, y)],
                    8,
                    color,
                    &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
                ))
                .unwrap();
        }
    }
}
