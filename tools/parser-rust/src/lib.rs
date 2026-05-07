use llama_simple_rust::ffi;
use std::ffi::{CStr, CString};
use std::fs;
use std::ptr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    Analysis,
    Template,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMessageType {
    None,
    ContentOnly,
    ReasoningContent,
    ToolCallOnly,
    ContentToolCall,
    ReasoningToolCall,
    ContentFakeToolCall,
    All,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DebugOptions {
    pub template_path: String,
    pub with_tools: bool,
    pub generation_prompt: bool,
    pub enable_reasoning: bool,
    pub debug_jinja: bool,
    pub force_tool_call: bool,
    pub mode: OutputMode,
    pub input_message: InputMessageType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisOptions {
    pub template_paths: Vec<String>,
    pub analyze_all: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateReport {
    pub bytes: usize,
    pub lines: usize,
    pub jinja_blocks: usize,
    pub jinja_expressions: usize,
    pub jinja_comments: usize,
    pub supports_tools: bool,
    pub supports_tool_calls: bool,
    pub mentions_reasoning: bool,
    pub mentions_generation_prompt: bool,
    pub mentions_system_role: bool,
    pub preserved_tokens: Vec<String>,
}

pub const ALL_TEMPLATE_PATHS: &[&str] = &[
    "models/templates/Apertus-8B-Instruct.jinja",
    "models/templates/Apriel-1.6-15b-Thinker-fixed.jinja",
    "models/templates/ByteDance-Seed-OSS.jinja",
    "models/templates/CohereForAI-c4ai-command-r-plus-tool_use.jinja",
    "models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja",
    "models/templates/GLM-4.6.jinja",
    "models/templates/GLM-4.7-Flash.jinja",
    "models/templates/Kimi-K2-Instruct.jinja",
    "models/templates/Kimi-K2-Thinking.jinja",
    "models/templates/MiMo-VL.jinja",
    "models/templates/MiniMax-M2.jinja",
    "models/templates/Mistral-Small-3.2-24B-Instruct-2506.jinja",
    "models/templates/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.jinja",
    "models/templates/NVIDIA-Nemotron-Nano-v2.jinja",
    "models/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja",
    "models/templates/NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja",
    "models/templates/Qwen-QwQ-32B.jinja",
    "models/templates/Qwen-Qwen2.5-7B-Instruct.jinja",
    "models/templates/Qwen3-Coder.jinja",
    "models/templates/deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja",
    "models/templates/deepseek-ai-DeepSeek-R1-Distill-Qwen-32B.jinja",
    "models/templates/deepseek-ai-DeepSeek-V3.1.jinja",
    "models/templates/fireworks-ai-llama-3-firefunction-v2.jinja",
    "models/templates/google-gemma-2-2b-it.jinja",
    "models/templates/ibm-granite-granite-3.3-2B-Instruct.jinja",
    "models/templates/llama-cpp-deepseek-r1.jinja",
    "models/templates/meetkai-functionary-medium-v3.1.jinja",
    "models/templates/meetkai-functionary-medium-v3.2.jinja",
    "models/templates/meta-llama-Llama-3.1-8B-Instruct.jinja",
    "models/templates/meta-llama-Llama-3.2-3B-Instruct.jinja",
    "models/templates/meta-llama-Llama-3.3-70B-Instruct.jinja",
    "models/templates/mistralai-Ministral-3-14B-Reasoning-2512.jinja",
    "models/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja",
    "models/templates/moonshotai-Kimi-K2.jinja",
    "models/templates/openai-gpt-oss-120b.jinja",
    "models/templates/unsloth-Apriel-1.5.jinja",
    "models/templates/unsloth-mistral-Devstral-Small-2507.jinja",
];

pub fn parse_debug_options<I, S>(args: I) -> Result<DebugOptions, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let args: Vec<String> = args.into_iter().map(Into::into).collect();
    if args.len() < 2 {
        return Err("missing template_or_gguf_path".to_string());
    }

    let mut opts = DebugOptions {
        template_path: args[1].clone(),
        with_tools: true,
        generation_prompt: true,
        enable_reasoning: true,
        debug_jinja: false,
        force_tool_call: false,
        mode: OutputMode::Both,
        input_message: InputMessageType::None,
    };

    for arg in &args[2..] {
        if arg == "--force-tool-call" {
            opts.force_tool_call = true;
        } else if arg == "--debug-jinja" {
            opts.debug_jinja = true;
        } else if arg == "--no-tools" {
            opts.with_tools = false;
        } else if let Some(value) = arg.strip_prefix("--generation-prompt=") {
            opts.generation_prompt = parse_bool_option(value);
        } else if let Some(value) = arg.strip_prefix("--enable-reasoning=") {
            opts.enable_reasoning = parse_bool_option(value);
        } else if let Some(value) = arg.strip_prefix("--output=") {
            opts.mode = match value {
                "analysis" => OutputMode::Analysis,
                "template" => OutputMode::Template,
                "both" => OutputMode::Both,
                _ => return Err(format!("unknown output mode: {value}")),
            };
        } else if let Some(value) = arg.strip_prefix("--input-message=") {
            opts.input_message = match value {
                "content_only" => InputMessageType::ContentOnly,
                "reasoning_content" => InputMessageType::ReasoningContent,
                "tool_call_only" => InputMessageType::ToolCallOnly,
                "content_tool_call" => InputMessageType::ContentToolCall,
                "reasoning_tool_call" => InputMessageType::ReasoningToolCall,
                "content_fake_tool_call" => InputMessageType::ContentFakeToolCall,
                "all" => InputMessageType::All,
                _ => return Err(format!("unknown input message type: {value}")),
            };
        } else {
            return Err(format!("unknown option: {arg}"));
        }
    }

    Ok(opts)
}

pub fn parse_analysis_options<I, S>(args: I) -> Result<AnalysisOptions, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let args: Vec<String> = args.into_iter().map(Into::into).collect();
    if args.len() < 2 {
        return Err("no templates specified".to_string());
    }

    let mut opts = AnalysisOptions {
        template_paths: Vec::new(),
        analyze_all: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--all" => {
                opts.analyze_all = true;
                i += 1;
            }
            "--template" => {
                let pattern = args
                    .get(i + 1)
                    .ok_or_else(|| "--template requires an argument".to_string())?
                    .to_lowercase();
                let mut found = false;
                for path in ALL_TEMPLATE_PATHS {
                    if path.to_lowercase().contains(&pattern) {
                        opts.template_paths.push((*path).to_string());
                        found = true;
                    }
                }
                if !found {
                    return Err(format!("no templates found matching: {pattern}"));
                }
                i += 2;
            }
            "--template-file" => {
                let path = args
                    .get(i + 1)
                    .ok_or_else(|| "--template-file requires an argument".to_string())?;
                opts.template_paths.push(path.clone());
                i += 2;
            }
            other => return Err(format!("unknown option: {other}")),
        }
    }

    if opts.analyze_all {
        opts.template_paths = ALL_TEMPLATE_PATHS
            .iter()
            .map(|p| (*p).to_string())
            .collect();
    }
    if opts.template_paths.is_empty() {
        return Err("no templates specified".to_string());
    }

    Ok(opts)
}

pub fn parse_bool_option(value: &str) -> bool {
    matches!(value, "1" | "true" | "yes")
}

pub fn read_template_source(path: &str) -> Result<String, String> {
    if path.ends_with(".gguf") {
        read_gguf_chat_template(path)
    } else {
        fs::read_to_string(path).map_err(|err| format!("could not open file {path}: {err}"))
    }
}

pub fn read_gguf_chat_template(path: &str) -> Result<String, String> {
    let c_path = CString::new(path).map_err(|_| "path contains interior NUL byte".to_string())?;
    let key = CString::new("tokenizer.chat_template").unwrap();
    let params = ffi::gguf_init_params {
        no_alloc: true,
        ctx: ptr::null_mut(),
    };

    unsafe {
        let ctx = ffi::gguf_init_from_file(c_path.as_ptr(), params);
        if ctx.is_null() {
            return Err(format!("could not open GGUF file: {path}"));
        }

        let key_id = ffi::gguf_find_key(ctx, key.as_ptr());
        if key_id < 0 {
            ffi::gguf_free(ctx);
            return Err("GGUF file does not contain tokenizer.chat_template".to_string());
        }

        let value = ffi::gguf_get_val_str(ctx, key_id);
        if value.is_null() {
            ffi::gguf_free(ctx);
            return Err("GGUF tokenizer.chat_template value is null".to_string());
        }

        let out = CStr::from_ptr(value).to_string_lossy().into_owned();
        ffi::gguf_free(ctx);
        Ok(out)
    }
}

pub fn analyze_template_source(source: &str) -> TemplateReport {
    let lower = source.to_lowercase();
    TemplateReport {
        bytes: source.len(),
        lines: source.lines().count(),
        jinja_blocks: source.matches("{%").count(),
        jinja_expressions: source.matches("{{").count(),
        jinja_comments: source.matches("{#").count(),
        supports_tools: lower.contains("tools"),
        supports_tool_calls: lower.contains("tool_calls") || lower.contains("tool_call"),
        mentions_reasoning: lower.contains("reasoning") || lower.contains("thinking"),
        mentions_generation_prompt: lower.contains("add_generation_prompt"),
        mentions_system_role: lower.contains("\"system\"")
            || lower.contains("'system'")
            || lower.contains("role == system")
            || lower.contains("role == 'system'")
            || lower.contains("role == \"system\""),
        preserved_tokens: find_preserved_tokens(source),
    }
}

pub fn render_static_scenario(source: &str, scenario: InputMessageType) -> String {
    let label = match scenario {
        InputMessageType::None => "none",
        InputMessageType::ContentOnly => "content_only",
        InputMessageType::ReasoningContent => "reasoning_content",
        InputMessageType::ToolCallOnly => "tool_call_only",
        InputMessageType::ContentToolCall => "content_tool_call",
        InputMessageType::ReasoningToolCall => "reasoning_tool_call",
        InputMessageType::ContentFakeToolCall => "content_fake_tool_call",
        InputMessageType::All => "all",
    };

    format!(
        "Scenario: {label}\nTemplate bytes: {}\nTemplate preview:\n{}",
        source.len(),
        source.chars().take(1200).collect::<String>()
    )
}

fn find_preserved_tokens(source: &str) -> Vec<String> {
    let candidates = [
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "<|assistant|>",
        "<|user|>",
        "<|system|>",
        "<|tool|>",
        "[INST]",
        "[/INST]",
    ];
    candidates
        .iter()
        .filter(|token| source.contains(**token))
        .map(|token| (*token).to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_debug_options() {
        let opts = parse_debug_options([
            "tool",
            "template.jinja",
            "--no-tools",
            "--generation-prompt=0",
            "--enable-reasoning=yes",
            "--output=analysis",
            "--input-message=all",
            "--force-tool-call",
        ])
        .unwrap();
        assert_eq!(opts.template_path, "template.jinja");
        assert!(!opts.with_tools);
        assert!(!opts.generation_prompt);
        assert!(opts.enable_reasoning);
        assert!(opts.force_tool_call);
        assert_eq!(opts.mode, OutputMode::Analysis);
        assert_eq!(opts.input_message, InputMessageType::All);
    }

    #[test]
    fn rejects_unknown_debug_mode() {
        let err = parse_debug_options(["tool", "template.jinja", "--output=json"]).unwrap_err();
        assert!(err.contains("unknown output mode"));
    }

    #[test]
    fn parses_template_pattern() {
        let opts = parse_analysis_options(["tool", "--template", "deepseek"]).unwrap();
        assert!(opts
            .template_paths
            .iter()
            .any(|path| path.to_lowercase().contains("deepseek")));
    }

    #[test]
    fn analyzes_template_features() {
        let report = analyze_template_source(
            "{% if tools %}{{ messages }}<tool_call>{{ add_generation_prompt }}{{ reasoning_content }}",
        );
        assert_eq!(report.jinja_blocks, 1);
        assert_eq!(report.jinja_expressions, 3);
        assert!(report.supports_tools);
        assert!(report.supports_tool_calls);
        assert!(report.mentions_reasoning);
        assert!(report.mentions_generation_prompt);
        assert_eq!(report.preserved_tokens, vec!["<tool_call>".to_string()]);
    }

    #[test]
    fn bool_parser_matches_original_truthy_values() {
        assert!(parse_bool_option("1"));
        assert!(parse_bool_option("true"));
        assert!(parse_bool_option("yes"));
        assert!(!parse_bool_option("0"));
        assert!(!parse_bool_option("false"));
        assert!(!parse_bool_option("anything-else"));
    }
}
