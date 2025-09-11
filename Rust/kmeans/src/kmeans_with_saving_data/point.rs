#[derive(Debug, Clone)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn zero() -> Self {
        Point { x: 0.0, y: 0.0 }
    }
    pub fn add(&self, other: &Point) -> Self {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
    pub fn div(&self, val: f64) -> Self {
        Point {
            x: self.x / val,
            y: self.y / val,
        }
    }
}

pub fn euclidean_distance(a: &Point, b: &Point) -> f64 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}
