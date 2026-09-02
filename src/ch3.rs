use core::f64;
use std::{fmt::Write, ops::Mul};

use nalgebra::{Matrix2, Vector2};
use plotters::prelude::*;
use serde::{Deserialize, Serialize};

// ---- HELPER METHODS ----
/// Computes vectorized softmax.
fn softmax_vec(x: &[f64]) -> Vec<f64> {
    let denom: f64 = x.iter().map(|x| f64::exp(*x)).sum();

    let result: Vec<f64> = x.iter().map(|x| f64::exp(*x) / denom).collect();

    result
}

/// Simple NN that is used in P3.10 and 3.17.
fn simple_neural_net(
    dataset: [(f64, &str); 4],
    mut weights: Matrix2<f64>,
    learning_rate: f64,
    iters: usize,
) {
    for i in 0..iters {
        let input = Vector2::new(dataset[i % 4].0, 1.0);
        let target = dataset[i % 4].1;

        // Compute forward pass
        let h_vec = weights.mul(input);
        let y_hat = Vector2::from_column_slice(&softmax_vec(h_vec.as_slice()));

        // Compute cross-entropy loss
        let mut loss = 0.0;
        if target == "Paris" {
            loss = -y_hat[0].ln();
        } else if target == "Berlin" || target == "Madrid" {
            loss = -y_hat[1].ln();
        };

        // Backprop
        let dL_dh1 = if target == "Paris" {
            y_hat[0] - 1.0
        } else {
            y_hat[0]
        };
        let dL_dh2 = if target == "Paris" {
            y_hat[1]
        } else {
            y_hat[1] - 1.0
        };
        let dh1_dm1 = input[0];
        let dh2_dm2 = input[0];
        let dh1_db1 = 1.0;
        let dh2_db2 = 1.0;

        let grad = Matrix2::new(
            dL_dh1 * dh1_dm1,
            dL_dh1 * dh1_db1,
            dL_dh2 * dh2_dm2,
            dL_dh2 * dh2_db2,
        );

        let mut accuracy = 0.0;
        for (longitude, city) in dataset.iter() {
            let x_i = Vector2::new(*longitude, 1.0);
            let h_i = weights.mul(x_i);
            let y_hat_i = Vector2::from_column_slice(&softmax_vec(h_i.as_slice()));
            let (max_row, _) = y_hat_i.argmax();

            if (max_row == 0 && *city == "Paris") || (max_row == 1 && (*city == "Berlin" || *city == "Madrid")) {
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
        writeln!(
            step_output,
            "y1       | {:.3}",
            if target == "Paris" { 1.0 } else { 0.0 }
        )
        .unwrap();
        writeln!(
            step_output,
            "y2       | {:.3}",
            if target == "Berlin" || target == "Madrid" { 1.0 } else { 0.0 }
        )
        .unwrap();
        writeln!(step_output, "loss     | {:.3}", loss).unwrap();
        writeln!(step_output, "accuracy | {:.3}", accuracy).unwrap();
        println!("{}", step_output);

        // Update the weights (doing it after printout so easier to check work)
        weights = weights - grad.mul(learning_rate);
    }
}

/// Single NN that's used in P3.23 and 3.29
///
/// Returns the step at which weights converged to within the target tolerance,
/// or `None` if it did not converge within `iters` steps.
fn single_neuron(
    dataset: [(f64, f64); 4],
    mut weights: Vector2<f64>,
    learning_rate: f64,
    iters: usize,
    loss_fn: &str,
    verbose: bool,
) -> Option<usize> {
    for i in 0..iters {
        let input = Vector2::new(dataset[i % dataset.len()].0, 1.0);
        let target = dataset[i % dataset.len()].1;

        // Forward pass
        let y_hat = weights.dot(&input);

        // Compute loss and backprop 
        let mut loss = 0.0;
        let mut dL_dm = 0.0;
        let mut dL_db = 0.0;
        if loss_fn == "l2" {
            loss = f64::powi(y_hat - target, 2);
            dL_dm = 2.0 * (y_hat - target) * input[0];
            dL_db = 2.0 * (y_hat - target);
        } else if loss_fn == "l1" {
            let diff = y_hat - target;
            loss = f64::abs(diff);
            if diff > 0.0 {
                dL_dm = input[0];
                dL_db = 1.0;
            } else if diff < 0.0 {
                dL_dm = -1.0 * input[0];
                dL_db = -1.0;
            }
        } 

        // Backprop
        let grad = Vector2::new(dL_dm, dL_db);

        // Print output
        if verbose {
            let mut step_output = String::new();
            writeln!(step_output, "Step     | {}", i).unwrap();
            writeln!(step_output, "x        | {:.3}", input[0]).unwrap();
            writeln!(step_output, "m        | {:.3}", weights[0]).unwrap();
            writeln!(step_output, "dL_dm    | {:.3}", grad[0]).unwrap();
            writeln!(step_output, "b        | {:.3}", weights[1]).unwrap();
            writeln!(step_output, "dL_db    | {:.3}", grad[1]).unwrap();
            writeln!(step_output, "y_hat    | {:.3}", y_hat).unwrap();
            writeln!(step_output, "y        | {:.3}", target).unwrap();
            writeln!(step_output, "loss     | {:.3}", loss).unwrap();
            println!("{}", step_output);
        }

        // Checking if converging to target
        let target_tol: f64 = 0.001;
        if f64::abs(weights[0] - 2.0) <= target_tol && f64::abs(weights[1] - 1.0) <= target_tol {
            if verbose {
                println!("Converged within {i} steps to be within +/-{target_tol:.3}");
                println!("m = {:.4}, b = {:.4}", weights[0], weights[1]);
            }
            return Some(i);
        }

        // Updating weights after printout
        weights = weights - learning_rate * grad;
    }

    None
}

// ---- PROBLEMS ----
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
        1.0, 0.0
    ); // Init values

    simple_neural_net(dataset, weights, learning_rate, 10);

    Some("P3.10 is finished".to_string())
}

/// Implement NN from P3.11 to P3.17
fn problem_3_17() -> Option<String> {
    let dataset: [(f64, &str); 4] = [
        // longitude, city
        (2.3514, "Paris"),
        (2.2945, "Paris"),
        (-3.7033, "Madrid"),
        (-3.6835, "Madrid"),
    ];

    let learning_rate: f64 = 0.1;

    let mut weights = Matrix2::new(
        -1.0, 0.0, 
        0.5, 0.0
    ); // Init values

    simple_neural_net(dataset, weights, learning_rate, 4);

    Some("P3.17 is finished".to_string())
}

/// Implement GD from 3.18 to 3.23
fn problem_3_23() -> Option<String> {
    let dataset: [(f64, f64); 4] = [
        (1.0, 3.0),
        (2.0, 5.0),
        (3.0, 7.0),
        (4.0, 9.0),
    ];

    let learning_rate: f64 = 0.1;
    let iters: usize = 8;
    let weights = Vector2::new(1.0, 0.0);    // [m, b]

    single_neuron(dataset, weights, learning_rate, iters, "l2", true);

    Some("P3.23 is finished".to_string())
}

#[derive(Serialize, Deserialize)]
struct LrSweepResult {
    learning_rate: f64,
    converged_step: Option<usize>,
}

/// Sweeps learning_rate from 0.01 to 0.15 (step 0.01) for the P3.23 single neuron
/// and records the step at which it converges to within the +/-0.001 tolerance.
/// Results are cached to a JSON file so the sweep doesn't need to be re-run, and a
/// PNG plot (learning rate vs. converged step) is generated from those results.
fn problem_3_23_learning_rate_sweep() -> Option<String> {
    let results_path = "results/p3_23_lr_sweep.json";
    let plot_path = "results/p3_23_lr_sweep.png";

    let results: Vec<LrSweepResult> = if let Ok(contents) = std::fs::read_to_string(results_path) {
        println!("Loading cached sweep results from {results_path}");
        serde_json::from_str(&contents).expect("failed to parse cached sweep results")
    } else {
        println!("No cached sweep results found, running sweep...");

        let dataset: [(f64, f64); 4] = [
            (1.0, 3.0),
            (2.0, 5.0),
            (3.0, 7.0),
            (4.0, 9.0),
        ];
        let iters: usize = 10000;

        let mut results = Vec::new();
        for lr_step in 1..=15 {
            let learning_rate = lr_step as f64 * 0.01;
            let weights = Vector2::new(1.0, 0.0);

            let converged_step = single_neuron(dataset, weights, learning_rate, iters, "l2", false);
            println!("learning_rate = {learning_rate:.2} | converged_step = {converged_step:?}");

            results.push(LrSweepResult { learning_rate, converged_step });
        }

        std::fs::create_dir_all("results").expect("failed to create results directory");
        let json = serde_json::to_string_pretty(&results).expect("failed to serialize sweep results");
        std::fs::write(results_path, json).expect("failed to write sweep results");
        println!("Saved sweep results to {results_path}");

        results
    };

    plot_lr_sweep(&results, plot_path).expect("failed to plot sweep results");
    println!("Saved sweep plot to {plot_path}");

    Some("P3.23 learning rate sweep is finished".to_string())
}

/// Plots learning rate (x-axis) vs. converged step (y-axis) to a PNG file.
fn plot_lr_sweep(results: &[LrSweepResult], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let points: Vec<(f64, usize)> = results
        .iter()
        .filter_map(|r| r.converged_step.map(|step| (r.learning_rate, step)))
        .collect();

    let max_step = points.iter().map(|(_, step)| *step).max().unwrap_or(0);
    let min_lr = results.iter().map(|r| r.learning_rate).fold(f64::INFINITY, f64::min);
    let max_lr = results.iter().map(|r| r.learning_rate).fold(f64::NEG_INFINITY, f64::max);

    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("P3.23: Converged Step vs. Learning Rate", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            (min_lr - 0.005)..(max_lr + 0.005),
            0..(max_step + max_step / 10 + 1),
        )?;

    chart
        .configure_mesh()
        .x_desc("Learning rate")
        .y_desc("Converged step")
        .draw()?;

    chart.draw_series(LineSeries::new(points.iter().copied(), &BLUE))?;
    chart.draw_series(
        points
            .iter()
            .map(|(lr, step)| Circle::new((*lr, *step), 4, BLUE.filled())),
    )?;

    root.present()?;

    for r in results {
        if r.converged_step.is_none() {
            println!("learning_rate {:.2} did not converge within the iteration cap", r.learning_rate);
        }
    }

    Ok(())
}

/// Implement GD from 3.24 to 3.29
fn problem_3_29() -> Option<String> {
    let dataset: [(f64, f64); 4] = [
        (1.0, 3.0),
        (2.0, 5.0),
        (3.0, 7.0),
        (4.0, 9.0),
    ];

    let learning_rate: f64 = 0.1;
    let iters: usize = 8;
    let weights = Vector2::new(1.0, 0.0);    // [m, b]

    single_neuron(dataset, weights, learning_rate, iters, "l1", true);

    Some("P3.29 is finished".to_string())
}

pub fn run() {
    println!("---- Chapter 3 Problems ----");
    // problem_3_1<<0();
    // println!("----");
    // problem_3_17();
    println!("");
    println!("---- P3.23 ----");
    problem_3_23();
    // println!("");
    // println!("---- P3.23 Learning Rate Sweep ----");
    // problem_3_23_learning_rate_sweep();
    // println!("");
    // println!("---- P3.29 ----");
    // problem_3_29();
}
