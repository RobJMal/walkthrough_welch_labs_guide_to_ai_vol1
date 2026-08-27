use std::{ops::Mul, fmt::Write};

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

    let learning_rate: f64 = 0.1;

    let mut weights = Matrix2::new(
        -1.0, 0.0, 
        1.0, 0.0,
    );  // Init values

    for i in 0..10 {
        let input = Vector2::new(dataset[i%4].0, 1.0);
        let target = dataset[i%4].1;

        // Compute forward pass
        let h_vec = weights.mul(input);
        let y_hat = Vector2::from_column_slice(
            &softmax_vec(h_vec.as_slice())
        );

        // Compute cross-entropy loss
        let mut loss = 0.0;
        if target == "Paris" {
            loss = -y_hat[0].ln();
        } else if target == "Berlin" {
            loss = -y_hat[1].ln();
        };

        // Backprop
        let dL_dh1 = if target == "Paris" {y_hat[0] - 1.0} else {y_hat[0]};
        let dL_dh2 = if target == "Paris" {y_hat[1]} else {y_hat[1] - 1.0};
        let dh1_dm1 = input[0];
        let dh2_dm2 = input[0];
        let dh1_db1 = 1.0;
        let dh2_db2 = 1.0;

        let grad = Matrix2::new(
            dL_dh1 * dh1_dm1, dL_dh1 * dh1_db1, 
            dL_dh2 * dh2_dm2, dL_dh2 * dh2_db2,
        );
        
        let mut accuracy = 0.0;
        for (longitude, city) in dataset.iter() {
            let x_i = Vector2::new(*longitude, 1.0);
            let h_i = weights.mul(x_i);
            let y_hat_i = Vector2::from_column_slice(
                &softmax_vec(h_i.as_slice())
            );
            let (max_row, _) = y_hat_i.argmax();
            
            if (max_row == 0 && *city == "Paris") || (max_row == 1 && *city == "Berlin") {
                accuracy += 1.0;
            }
        }
        accuracy /= dataset.len() as f64;

        let mut step_output = String::new();
        writeln!(step_output, "Step     | {}", i).unwrap();
        writeln!(step_output, "x        | {:.3}", input[0]).unwrap();
        writeln!(step_output, "m1       | {:.3}", weights[0]).unwrap();
        writeln!(step_output, "m2       | {:.3}", weights[1]).unwrap();
        writeln!(step_output, "dL/dm1   | {:.3}", grad[0]).unwrap();
        writeln!(step_output, "dL/dm2   | {:.3}", grad[1]).unwrap();
        writeln!(step_output, "b1       | {:.3}", weights[2]).unwrap();
        writeln!(step_output, "b2       | {:.3}", weights[3]).unwrap();
        writeln!(step_output, "dL/db1   | {:.3}", grad[2]).unwrap();
        writeln!(step_output, "dL/db2   | {:.3}", grad[3]).unwrap();
        writeln!(step_output, "h1       | {:.3}", h_vec[0]).unwrap();
        writeln!(step_output, "h2       | {:.3}", h_vec[1]).unwrap();
        writeln!(step_output, "y_hat1   | {:.3}", y_hat[0]).unwrap();
        writeln!(step_output, "y_hat2   | {:.3}", y_hat[1]).unwrap();
        writeln!(step_output, "y1       | {:.3}", if target == "Paris" {1.0} else {0.0}).unwrap();
        writeln!(step_output, "y2       | {:.3}", if target == "Berlin" {1.0} else {0.0}).unwrap();
        writeln!(step_output, "loss     | {:.3}", loss).unwrap();
        writeln!(step_output, "accuracy | {:.3}", accuracy).unwrap();
        println!("{}", step_output);

        // Update the weights (doing it after printout so easier to check work)
        weights = weights - grad.mul(learning_rate);
    }


    Some("problem is finished".to_string())
}

pub fn run() {
    println!("---- Chapter 3 Problems ----");
    problem_3_10();
}