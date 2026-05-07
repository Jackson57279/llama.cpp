use std::env;

fn main() {
    let path = env::args().next().unwrap_or_else(|| "main".to_string());
    print!("{}", mtmd_deprecation_warning::warning_message(&path));
    std::process::exit(1);
}
