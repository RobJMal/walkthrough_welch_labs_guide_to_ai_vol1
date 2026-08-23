use nalgebra::Vector3;

/// Problem 1.16: Implement Perceptron Learning Rule (PLR)
/// 
/// Test on example in Exercise 1.4
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

fn main() {
    println!("---- Chapter 1 Problems ----");
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
}
