use llama_parser_tools_rust::{
    analyze_template_source, parse_analysis_options, read_template_source, ALL_TEMPLATE_PATHS,
};
use std::env;

fn print_usage(program: &str) {
    eprintln!("Usage: {program} [options]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --template <name>       Analyze specific template from test suite");
    eprintln!("  --template-file <path>  Analyze custom template file");
    eprintln!("  --all                   Analyze all templates from test suite");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let opts = match parse_analysis_options(args.iter().cloned()) {
        Ok(opts) => opts,
        Err(err) => {
            eprintln!("Error: {err}");
            print_usage(
                args.first()
                    .map(String::as_str)
                    .unwrap_or("llama-template-analysis"),
            );
            std::process::exit(1);
        }
    };

    eprintln!();
    eprintln!("================================================================================");
    eprintln!("                      TEMPLATE ANALYSIS TOOL");
    eprintln!("================================================================================");
    eprintln!(
        "Analyzing {} template(s) (known templates: {})",
        opts.template_paths.len(),
        ALL_TEMPLATE_PATHS.len()
    );

    for path in &opts.template_paths {
        eprintln!();
        eprintln!(
            "================================================================================"
        );
        eprintln!("                    ANALYZING TEMPLATE: {path}");
        eprintln!(
            "================================================================================"
        );

        match read_template_source(path) {
            Ok(source) => {
                let report = analyze_template_source(&source);
                eprintln!("bytes: {}", report.bytes);
                eprintln!("lines: {}", report.lines);
                eprintln!("jinja_blocks: {}", report.jinja_blocks);
                eprintln!("jinja_expressions: {}", report.jinja_expressions);
                eprintln!("jinja_comments: {}", report.jinja_comments);
                eprintln!("supports_tools: {}", report.supports_tools);
                eprintln!("supports_tool_calls: {}", report.supports_tool_calls);
                eprintln!("mentions_reasoning: {}", report.mentions_reasoning);
                eprintln!(
                    "mentions_generation_prompt: {}",
                    report.mentions_generation_prompt
                );
                eprintln!("mentions_system_role: {}", report.mentions_system_role);
                eprintln!("preserved_tokens: {}", report.preserved_tokens.join(", "));
            }
            Err(err) => eprintln!("Error reading template: {err}"),
        }
    }

    eprintln!();
    eprintln!("================================================================================");
    eprintln!("                      ANALYSIS COMPLETE");
    eprintln!("================================================================================");
}
