use nalgebra::Vector3;

/// Problem 1.16: Implement Perceptron Learning Rule (PLR)
fn problem_1_16(input_targets: &[(Vector3<i32>, i32); 8]) -> std::option::Option<String> {
    // Params of model
    let mut weights = Vector3::new(0, 0, 0);

    for (i, (input, target)) in input_targets.iter().enumerate() {
        let y_hat = weights.dot(&input);

        // --- Perceptron Learning Rule ----
        // Case 1: target is positive, output is negative
        if *target > 0 && y_hat <= 0 {
            weights = weights + input;
        }
        // Case 2: target is negative, output is positive 
        else if *target < 0 && y_hat >= 0 {
            weights = weights - input;
        }
        // Case 3 & 4: target and output are the same signs -> DO nothing
        println!("Step {}", i+1);
        println!("target = {target} |  y_hat = {y_hat}");
        println!("Updated parameters | w1: {}, w2: {}, b: {}", weights[0], weights[1], weights[2]);
        println!("");
    }

    Some("problem is correct".to_string())
}

/// Problem 1.17: Implement Least Mean Squares (LMS)
fn problem_1_17(input_targets: &[(Vector3<f64>, f64); 12]) -> Option<String> {
    let alpha: f64 = 0.2;   // learning rate
    let mut weights = Vector3::new(0.0, 0.0, 0.0);

    // --- LMS ---
    for (i, (input, target)) in input_targets.iter().enumerate() {
        let y_hat = weights.dot(input);
        let diff = target - y_hat;
        let error = f64::powi(diff, 2);

        let dE = -2.0 * diff * input;
        let weight_update_term = (alpha / 2.0) * dE;
        weights = weights - weight_update_term;

        let result: String = format!(
            "Step {} | y-y_hat: {diff:.2} | E: {error:.2} | \
             dE/dw1: {:.2} | dE/dw2: {:.2} | dE/db: {:.2} | \
             (a/2)dE/dw1: {:.2} | (a/2)dE/dw2: {:.2} | (a/2)dE/db: {:.2}", 
            i + 1, 
            dE[0], dE[1], dE[2], 
            weight_update_term[0], weight_update_term[1], weight_update_term[2]
        );
        println!("{}", result);
    }

    Some("Problem is correct".to_string())
}

fn main() {
    println!("---- Chapter 1 Problems ----");

    // ---- P1.16 ----
    println!("---- Problem 1.16 ----");
    let test_01_p1_4: [(Vector3<i32>, i32); 8] = [  
        // (x1, x2, xb), xb always = 1 since "always on"
        (Vector3::new(-1, -1, 1), -1),
        (Vector3::new(-1, 1, 1), -1),
        (Vector3::new(1, -1, 1), 1),
        (Vector3::new(1, 1, 1), 1),
        (Vector3::new(-1, -1, 1), -1),
        (Vector3::new(-1, 1, 1), -1),
        (Vector3::new(1, -1, 1), 1),
        (Vector3::new(1, 1, 1), 1),
    ];
    problem_1_16(&test_01_p1_4);

    // ---- P1.17 ----
    println!("---- Problem 1.17 ----");
    let test_01_p1_10: [(Vector3<f64>, f64); 12] = [  
        // (x1, x2, xb), xb always = 1 since "always on"
        (Vector3::new(-1.0, -1.0, 1.0), -1.0),
        (Vector3::new(-1.0, 1.0, 1.0), -1.0),
        (Vector3::new(1.0, -1.0, 1.0), 1.0),
        (Vector3::new(1.0, 1.0, 1.0), 1.0),
        (Vector3::new(-1.0, -1.0, 1.0), -1.0),
        (Vector3::new(-1.0, 1.0, 1.0), -1.0),
        (Vector3::new(1.0, -1.0, 1.0), 1.0),
        (Vector3::new(1.0, 1.0, 1.0), 1.0),
        (Vector3::new(-1.0, -1.0, 1.0), -1.0),
        (Vector3::new(-1.0, 1.0, 1.0), -1.0),
        (Vector3::new(1.0, -1.0, 1.0), 1.0),
        (Vector3::new(1.0, 1.0, 1.0), 1.0),
    ];
    problem_1_17(&test_01_p1_10);
}
