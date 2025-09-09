#[derive(Debug, Clone)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

pub fn euclidean_distance(a: &Point, b: &Point) -> f64 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

pub fn mean(points: &[Point]) -> Point {
    let n = points.len() as f64;
    let (sum_x, sum_y) = points.iter().fold((0.0, 0.0), |(sx, sy), p| (sx + p.x, sy + p.y));
    Point { x: sum_x / n, y: sum_y / n }
}
