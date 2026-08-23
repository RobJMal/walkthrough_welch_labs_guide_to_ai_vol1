use nalgebra::Vector3;

/// Problem 1.16: Implement Perceptron Learning Rule (PLR)
/// 
/// Test on example in Exercise 1.4
fn problem_1_16() -> std::option::Option<String> {
    // Init values based on problem statement
    let input_step_1 = Vector3::new(-1, -1, 1);
    let target_step_1: i32 = -1; 
    let w_step_1 = Vector3::new(0, 0, 0);
    let y_hat_step_1 = w_step_1.dot(&input_step_1);

    // Updated values
    let mut weights = w_step_1.clone();

    println!("target = {target_step_1}, y_hat = {y_hat_step_1}");

    // Case 1: target is positive, output is negative
    if target_step_1 > 1 && y_hat_step_1 <= 0 {
        weights = weights + input_step_1;
    }
    // Case 2: target is negative, output is positive 
    else if target_step_1 < 0 && y_hat_step_1 >= 0 {
        weights = weights - input_step_1;
    }
    // Case 3 & 4: target and output are the same signs -> DO nothing

    println!("w = {weights}");

    Some("problem is correct".to_string())
}

fn main() {
    println!("---- Chapter 1 Problems ----");
    problem_1_16();
}
