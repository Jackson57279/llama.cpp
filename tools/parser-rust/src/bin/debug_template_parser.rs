use llama_parser_tools_rust::{
    analyze_template_source, parse_debug_options, read_template_source, render_static_scenario,
    InputMessageType, OutputMode,
};
use std::env;

fn print_usage(program: &str) {
    eprintln!("Usage: {program} <template_or_gguf_path> [options]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --no-tools              Disable tool definitions");
    eprintln!("  --force-tool-call       Set tool calls to forced");
    eprintln!("  --generation-prompt=0|1 Set add_generation_prompt (default: 1)");
    eprintln!("  --enable-reasoning=0|1  Enable reasoning parsing (default: 1)");
    eprintln!("  --output=MODE           Output mode: analysis, template, both (default: both)");
    eprintln!("  --debug-jinja           Accepted for compatibility");
    eprintln!("  --input-message=TYPE    content_only, reasoning_content, tool_call_only,");
    eprintln!("                          content_tool_call, reasoning_tool_call,");
    eprintln!("                          content_fake_tool_call, all");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let opts = match parse_debug_options(args.iter().cloned()) {
        Ok(opts) => opts,
        Err(err) => {
            eprintln!("Error: {err}");
            print_usage(
                args.first()
                    .map(String::as_str)
                    .unwrap_or("llama-debug-template-parser"),
            );
            std::process::exit(1);
        }
    };

    let source = match read_template_source(&opts.template_path) {
        Ok(source) => source,
        Err(err) => {
            eprintln!("Error reading template: {err}");
            std::process::exit(1);
        }
    };

    eprintln!("Analyzing template: {}", opts.template_path);
    eprintln!(
        "Options: with_tools={}, generation_prompt={}, enable_reasoning={}, force_tool_call={}",
        opts.with_tools, opts.generation_prompt, opts.enable_reasoning, opts.force_tool_call
    );

    if matches!(opts.mode, OutputMode::Template | OutputMode::Both)
        && opts.input_message != InputMessageType::None
    {
        eprintln!();
        eprintln!(
            "================================================================================"
        );
        eprintln!("                         TEMPLATE STATIC OUTPUT");
        eprintln!(
            "================================================================================"
        );
        eprintln!("{}", render_static_scenario(&source, opts.input_message));
    }

    if matches!(opts.mode, OutputMode::Analysis | OutputMode::Both) {
        let report = analyze_template_source(&source);
        eprintln!();
        eprintln!(
            "================================================================================"
        );
        eprintln!("                           TEMPLATE ANALYSIS");
        eprintln!(
            "================================================================================"
        );
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
}
