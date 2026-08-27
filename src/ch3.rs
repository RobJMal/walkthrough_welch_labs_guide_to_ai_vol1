use std::ops::Mul;

use nalgebra::{Matrix2, Vector2};

// ---- HELPER METHODS ----
/// Computes vectorized softmax. 
fn softmax_vec(x: &[f32]) -> Vec<f32> {
    let denom: f32 = x
        .iter()
        .map(|x| f32::exp(*x))
        .sum();

    let result: Vec<f32> = x.iter()
        .map(|x| f32::exp(*x) / denom)
        .collect(); 

    result
}

/// Implement neural network from P3.4 to P3.10. 
fn problem_3_10() -> Option<String> {
    let dataset: [(f32, &str); 4] = [
        // longitude, city
        (2.3514, "Paris"),
        (2.2945, "Paris"),
        (13.4050, "Berlin"),
        (13.3777, "Berlin"),
    ];

    let learning_rate: f32 = 0.1;

    let mut weights = Matrix2::new(
        -1.0, 0.0, 
        1.0, 0.0,
    );  // Init values

    let input = Vector2::new(dataset[0].0, 1.0);
    let target = dataset[0].1;

    // Compute forward pass
    let h_vec = weights.mul(input);
    let y_hat = Vector2::from_column_slice(
        &softmax_vec(h_vec.as_slice())
    );
    println!("h_vec: {}", h_vec);
    println!("y_hat: {:?}", y_hat);

    // Compute cross-entropy loss
    let mut loss = 0.0;
    if target == "Paris" {
        loss = -y_hat[0].ln();
    } else if target == "Berlin" {
        loss = -y_hat[1].ln();
    };

    // Backprop
    let dL_dh1 = y_hat[0] - 1.0;
    let dL_dh2 = y_hat[1];
    let dh1_dm1 = input[0];
    let dh2_dm2 = input[0];
    let dh1_db1 = 1.0;
    let dh2_db2 = 1.0;

    let grad = Matrix2::new(
        dL_dh1 * dh1_dm1, dL_dh1 * dh1_db1, 
        dL_dh2 * dh2_dm2, dL_dh2 * dh2_db2,
    );
    weights = weights - grad.mul(learning_rate);
    
    println!("weights: {weights}");

    Some("problem is finished".to_string())
}

pub fn run() {
    println!("---- Chapter 3 Problems ----");
    problem_3_10();
}