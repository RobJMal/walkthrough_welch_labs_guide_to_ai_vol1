use std::ops::Mul;

use nalgebra::{Matrix2, Vector2};

// ---- HELPER METHODS ----
/// Computes vectorized softmax. 
fn softmax_vec(x: &[f64]) -> Vec<f64> {
    let denom: f64 = x
        .iter()
        .map(|x| f64::exp(*x))
        .sum();

    let result: Vec<f64> = x.iter()
        .map(|x| f64::exp(*x) / denom)
        .collect(); 

    result
}

/// Implement neural network from P3.4 to P3.10. 
fn problem_3_10() -> Option<String> {
    let dataset: [(f64, &str); 4] = [
        // longitude, city
        (2.3514, "Paris"),
        (2.2945, "Paris"),
        (13.4050, "Berlin"),
        (13.3777, "Berlin"),
    ];

    let mut weights = Matrix2::new(
        -1.0, 0.0, 
        1.0, 0.0,
    );  // Init values

    let input = Vector2::new(dataset[0].0, 1.0);

    // Compute forward pass
    let h_vec = weights.mul(input);
    let y_hat = Vector2::from_column_slice(
        &softmax_vec(h_vec.as_slice())
    );
    println!("h_vec: {}", h_vec);
    println!("y_hat: {:?}", y_hat);

    Some("problem is finished".to_string())
}

pub fn run() {
    println!("---- Chapter 3 Problems ----");
    problem_3_10();
}