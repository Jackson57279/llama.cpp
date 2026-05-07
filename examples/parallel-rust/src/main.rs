use llama_simple_rust::ffi;
use std::env;
use std::ffi::CString;
use std::ptr;

const SYSTEM_PROMPT: &str = r#"Transcript of a never ending dialog, where the User interacts with an Assistant.
The Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User:
Recommend a nice restaurant in the area.
Assistant:
I recommend the restaurant "The Golden Duck". It is a 5 star restaurant with a great view of the city. The food is delicious and the service is excellent. The prices are reasonable and the portions are generous. The restaurant is located at 123 Main Street, New York, NY 10001. The phone number is (212) 555-1234. The hours are Monday through Friday from 11:00 am to 10:00 pm. The restaurant is closed on Saturdays and Sundays.
User:
Who is Richard Feynman?
Assistant:
Richard Feynman was an American physicist who is best known for his work in quantum mechanics and particle physics. He was awarded the Nobel Prize in Physics in 1965 for his contributions to the development of quantum electrodynamics. He was a popular lecturer and author, and he wrote several books, including "Surely You're Joking, Mr. Feynman!" and "What Do You Care What Other People Think?".
"#;

const QUESTIONS: &[&str] = &[
    "What is the tallest mountain in the world?",
    "Who was the first person to win two Nobel Prizes?",
    "Which country invented paper?",
    "What organ is primarily responsible for pumping blood throughout the body?",
    "Which planet is known for its prominent ring system?",
    "Who directed the movie 'Inception'?",
    "What is the freezing point of water in Fahrenheit?",
    "Which animal is known to have the longest lifespan?",
    "What language has the most native speakers worldwide?",
    "What is the capital city of Canada?",
    "Who is credited with inventing the World Wide Web?",
    "Which metal is liquid at room temperature?",
    "What is the term for an animal that eats both plants and meat?",
    "Who painted 'The Starry Night'?",
    "What gas do humans exhale that plants use for photosynthesis?",
    "What year did World War II end?",
    "Which continent has the most countries?",
    "Who wrote the novel 'Frankenstein'?",
    "What does DNA stand for?",
    "What is the main ingredient in traditional Japanese miso soup?",
];

const ANSWERS: &[&str] = &[
    "The tallest mountain in the world is Mount Everest.",
    "Marie Curie was the first person to win two Nobel Prizes.",
    "Paper was invented in China.",
    "The heart is the organ responsible for pumping blood.",
    "Saturn is known for its prominent ring system.",
    "Christopher Nolan directed the movie 'Inception'.",
    "The freezing point of water in Fahrenheit is 32 degrees F.",
    "The bowhead whale is known to have the longest lifespan among mammals.",
    "Mandarin Chinese has the most native speakers in the world.",
    "The capital city of Canada is Ottawa.",
    "Tim Berners-Lee is credited with inventing the World Wide Web.",
    "Mercury is the metal that is liquid at room temperature.",
    "An animal that eats both plants and meat is called an omnivore.",
    "'The Starry Night' was painted by Vincent van Gogh.",
    "Humans exhale carbon dioxide, which plants use in photosynthesis.",
    "World War II ended in 1945.",
    "Africa is the continent with the most countries.",
    "The novel 'Frankenstein' was written by Mary Shelley.",
    "DNA stands for Deoxyribonucleic Acid.",
    "The main ingredient in traditional Japanese miso soup is fermented soybean paste.",
];

const DEFAULT_PROMPTS: &[&str] = &[
    "What is the meaning of life?",
    "Tell me an interesting fact about llamas.",
    "What is the best way to cook a steak?",
    "Are you familiar with the Special Theory of Relativity and can you explain it to me?",
    "Recommend some interesting books to read.",
    "What is the best way to learn a new language?",
    "How to get a job at Google?",
    "If you could have any superpower, what would it be?",
    "I want to learn how to play the piano. What would be the best way to do it?",
];

#[derive(Debug, Clone, PartialEq)]
pub struct Args {
    pub model_path: String,
    pub prompt: String,
    pub prompt_file: String,
    pub n_predict: i32,
    pub n_parallel: i32,
    pub n_sequences: i32,
    pub cont_batching: bool,
    pub is_pp_shared: bool,
    pub n_junk: i32,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_gpu_layers: i32,
    pub top_k: i32,
    pub top_p: f32,
    pub temp: f32,
    pub seed: i32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            prompt: String::new(),
            prompt_file: String::new(),
            n_predict: 128,
            n_parallel: 1,
            n_sequences: 1,
            cont_batching: true,
            is_pp_shared: false,
            n_junk: 1,
            n_ctx: 512,
            n_batch: 512,
            n_gpu_layers: 99,
            top_k: 40,
            top_p: 0.95,
            temp: 0.8,
            seed: ffi::LLAMA_DEFAULT_SEED as i32,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    MissingModel,
    MissingValue(String),
    InvalidInteger(String, String),
    InvalidFloat(String, String),
    InvalidValue(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::MissingModel => write!(f, "missing required -m/--model model.gguf"),
            ParseError::MissingValue(flag) => write!(f, "missing value for {flag}"),
            ParseError::InvalidInteger(flag, value) => {
                write!(f, "invalid integer for {flag}: {value}")
            }
            ParseError::InvalidFloat(flag, value) => write!(f, "invalid float for {flag}: {value}"),
            ParseError::InvalidValue(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for ParseError {}

pub fn parse_args<I, S>(args: I) -> Result<Args, ParseError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut parsed = Args::default();
    let mut iter = args.into_iter().map(Into::into);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-m" | "--model" => parsed.model_path = take(&mut iter, &arg)?,
            "-p" | "--prompt" => parsed.prompt = take(&mut iter, &arg)?,
            "-f" | "--file" | "--prompt-file" => {
                parsed.prompt_file = take(&mut iter, &arg)?;
                parsed.prompt = std::fs::read_to_string(&parsed.prompt_file).map_err(|err| {
                    ParseError::InvalidValue(format!(
                        "failed to read prompt file {}: {err}",
                        parsed.prompt_file
                    ))
                })?;
            }
            "-n" | "--n-predict" => parsed.n_predict = parse_i32(&mut iter, &arg)?,
            "-np" | "--parallel" => parsed.n_parallel = parse_i32(&mut iter, &arg)?,
            "-ns" | "--sequences" => parsed.n_sequences = parse_i32(&mut iter, &arg)?,
            "--cont-batching" => parsed.cont_batching = parse_bool(&mut iter, &arg)?,
            "--no-cont-batching" => parsed.cont_batching = false,
            "-pps" | "--parallel-prompt-shared" => parsed.is_pp_shared = true,
            "--junk" => parsed.n_junk = parse_i32(&mut iter, &arg)?,
            "-c" | "--ctx-size" => parsed.n_ctx = parse_u32(&mut iter, &arg)?,
            "-b" | "--batch-size" => parsed.n_batch = parse_u32(&mut iter, &arg)?,
            "-ngl" | "--gpu-layers" => parsed.n_gpu_layers = parse_i32(&mut iter, &arg)?,
            "--top-k" => parsed.top_k = parse_i32(&mut iter, &arg)?,
            "--top-p" => parsed.top_p = parse_f32(&mut iter, &arg)?,
            "--temp" => parsed.temp = parse_f32(&mut iter, &arg)?,
            "-s" | "--seed" => parsed.seed = parse_i32(&mut iter, &arg)?,
            "-h" | "--help" => return Err(ParseError::MissingModel),
            _ => {}
        }
    }

    if parsed.model_path.is_empty() {
        return Err(ParseError::MissingModel);
    }
    if parsed.n_parallel <= 0 {
        return Err(ParseError::InvalidValue(
            "-np/--parallel must be positive".to_string(),
        ));
    }
    if parsed.n_sequences <= 0 {
        return Err(ParseError::InvalidValue(
            "-ns/--sequences must be positive".to_string(),
        ));
    }
    if parsed.n_ctx == 0 || parsed.n_batch == 0 {
        return Err(ParseError::InvalidValue(
            "context and batch sizes must be positive".to_string(),
        ));
    }
    parsed.n_junk = parsed.n_junk.max(1);
    Ok(parsed)
}

fn take<I>(iter: &mut I, flag: &str) -> Result<String, ParseError>
where
    I: Iterator<Item = String>,
{
    iter.next()
        .ok_or_else(|| ParseError::MissingValue(flag.to_string()))
}

fn parse_i32<I>(iter: &mut I, flag: &str) -> Result<i32, ParseError>
where
    I: Iterator<Item = String>,
{
    let value = take(iter, flag)?;
    value
        .parse()
        .map_err(|_| ParseError::InvalidInteger(flag.to_string(), value))
}

fn parse_u32<I>(iter: &mut I, flag: &str) -> Result<u32, ParseError>
where
    I: Iterator<Item = String>,
{
    let value = take(iter, flag)?;
    value
        .parse()
        .map_err(|_| ParseError::InvalidInteger(flag.to_string(), value))
}

fn parse_f32<I>(iter: &mut I, flag: &str) -> Result<f32, ParseError>
where
    I: Iterator<Item = String>,
{
    let value = take(iter, flag)?;
    value
        .parse()
        .map_err(|_| ParseError::InvalidFloat(flag.to_string(), value))
}

fn parse_bool<I>(iter: &mut I, flag: &str) -> Result<bool, ParseError>
where
    I: Iterator<Item = String>,
{
    match take(iter, flag)?.as_str() {
        "1" | "true" | "on" | "yes" => Ok(true),
        "0" | "false" | "off" | "no" => Ok(false),
        value => Err(ParseError::InvalidValue(format!(
            "invalid boolean for {flag}: {value}"
        ))),
    }
}

pub fn trim_ascii(text: &str) -> String {
    text.trim_matches(|c: char| c.is_ascii_whitespace())
        .to_string()
}

pub fn split_prompt_lines(input: &str) -> Vec<String> {
    input.lines().map(str::to_string).collect()
}

#[derive(Debug, Clone)]
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.state / 65536) % 32768
    }

    fn next_usize(&mut self, limit: usize) -> usize {
        if limit == 0 {
            0
        } else {
            self.next() as usize % limit
        }
    }
}

struct Model(*mut ffi::llama_model);

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                ffi::llama_model_free(self.0);
            }
        }
    }
}

struct Context(*mut ffi::llama_context);

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                ffi::llama_free(self.0);
            }
        }
    }
}

struct Batch(ffi::llama_batch);

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe {
            ffi::llama_batch_free(self.0);
        }
    }
}

struct Sampler(*mut ffi::llama_sampler);

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                ffi::llama_sampler_free(self.0);
            }
        }
    }
}

struct Client {
    id: i32,
    seq_id: i32,
    sampled: ffi::llama_token,
    t_start_prompt: i64,
    t_start_gen: i64,
    n_past: i32,
    n_prompt: i32,
    n_decoded: i32,
    i_batch: i32,
    input: String,
    prompt: String,
    response: String,
    sampler: Sampler,
}

struct Backend;

impl Backend {
    fn init() -> Self {
        unsafe {
            ffi::ggml_backend_load_all();
            ffi::llama_backend_init();
            ffi::llama_numa_init(0);
        }
        Self
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        unsafe {
            ffi::llama_backend_free();
        }
    }
}

fn sampler_init(args: &Args, seed: i32) -> Sampler {
    unsafe {
        let chain = ffi::llama_sampler_chain_init(ffi::llama_sampler_chain_default_params());
        ffi::llama_sampler_chain_add(chain, ffi::llama_sampler_init_top_k(args.top_k));
        ffi::llama_sampler_chain_add(chain, ffi::llama_sampler_init_top_p(args.top_p, 1));
        ffi::llama_sampler_chain_add(chain, ffi::llama_sampler_init_temp(args.temp));
        ffi::llama_sampler_chain_add(chain, ffi::llama_sampler_init_dist(seed as u32));
        Sampler(chain)
    }
}

fn tokenize(
    vocab: *const ffi::llama_vocab,
    text: &str,
    add_special: bool,
    parse_special: bool,
) -> Result<Vec<ffi::llama_token>, String> {
    let text_c =
        CString::new(text).map_err(|_| "text contains an interior NUL byte".to_string())?;
    let n_tokens = unsafe {
        -ffi::llama_tokenize(
            vocab,
            text_c.as_ptr(),
            text.len() as i32,
            ptr::null_mut(),
            0,
            add_special,
            parse_special,
        )
    };
    if n_tokens <= 0 {
        return Err("failed to size tokenization".to_string());
    }
    let mut tokens = vec![0_i32; n_tokens as usize];
    let n = unsafe {
        ffi::llama_tokenize(
            vocab,
            text_c.as_ptr(),
            text.len() as i32,
            tokens.as_mut_ptr(),
            tokens.len() as i32,
            add_special,
            parse_special,
        )
    };
    if n < 0 {
        return Err("failed to tokenize text".to_string());
    }
    tokens.truncate(n as usize);
    Ok(tokens)
}

fn token_to_piece(
    vocab: *const ffi::llama_vocab,
    token: ffi::llama_token,
) -> Result<String, String> {
    let mut buf = vec![0_i8; 128];
    let n = unsafe {
        ffi::llama_token_to_piece(vocab, token, buf.as_mut_ptr(), buf.len() as i32, 0, true)
    };
    if n < 0 {
        return Err("failed to convert token to piece".to_string());
    }
    let bytes = buf[..n as usize]
        .iter()
        .map(|&c| c as u8)
        .collect::<Vec<_>>();
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn batch_clear(batch: &mut ffi::llama_batch) {
    batch.n_tokens = 0;
}

unsafe fn batch_add(
    batch: &mut ffi::llama_batch,
    token: ffi::llama_token,
    pos: ffi::llama_pos,
    seq_ids: &[ffi::llama_seq_id],
    logits: bool,
) {
    let i = batch.n_tokens as isize;
    *batch.token.offset(i) = token;
    *batch.pos.offset(i) = pos;
    *batch.n_seq_id.offset(i) = seq_ids.len() as i32;
    let seq_slot = *batch.seq_id.offset(i);
    for (j, seq_id) in seq_ids.iter().enumerate() {
        *seq_slot.add(j) = *seq_id;
    }
    *batch.logits.offset(i) = if logits { 1 } else { 0 };
    batch.n_tokens += 1;
}

unsafe fn set_last_batch_logits(batch: &mut ffi::llama_batch) {
    if batch.n_tokens > 0 {
        *batch.logits.offset(batch.n_tokens as isize - 1) = 1;
    }
}

unsafe fn batch_view(batch: &ffi::llama_batch, start: i32, n_tokens: i32) -> ffi::llama_batch {
    let offset = start as isize;
    ffi::llama_batch {
        n_tokens,
        token: batch.token.offset(offset),
        embd: ptr::null_mut(),
        pos: batch.pos.offset(offset),
        n_seq_id: batch.n_seq_id.offset(offset),
        seq_id: batch.seq_id.offset(offset),
        logits: batch.logits.offset(offset),
    }
}

fn build_prompt(
    is_shared: bool,
    n_tokens_system: i32,
    input: &str,
    n_junk: i32,
    rng: &mut Lcg,
) -> (String, i32, i32) {
    let mut n_past = 0;
    let mut prompt = String::new();
    if is_shared {
        n_past = n_tokens_system;
    } else {
        prompt.push_str(SYSTEM_PROMPT);
    }
    let n_junk_cur = rng.next_usize(n_junk.max(1) as usize) as i32;
    for _ in 0..n_junk_cur {
        let r = rng.next_usize(QUESTIONS.len());
        prompt.push_str("User:\n");
        prompt.push_str(QUESTIONS[r]);
        prompt.push_str("\nAssistant:\n ");
        prompt.push_str(ANSWERS[r]);
        prompt.push('\n');
    }
    prompt.push_str("User:\n");
    prompt.push_str(input);
    prompt.push_str("\nAssistant:\n");
    (prompt, n_past, n_junk_cur)
}

fn run(args: Args) -> Result<(), String> {
    let _backend = Backend::init();
    let model_path = CString::new(args.model_path.as_str())
        .map_err(|_| "model path contains an interior NUL byte".to_string())?;

    let mut model_params = unsafe { ffi::llama_model_default_params() };
    model_params.n_gpu_layers = args.n_gpu_layers;
    let model =
        Model(unsafe { ffi::llama_model_load_from_file(model_path.as_ptr(), model_params) });
    if model.0.is_null() {
        return Err("unable to load model".to_string());
    }

    let mut ctx_params = unsafe { ffi::llama_context_default_params() };
    ctx_params.n_ctx = args.n_ctx;
    ctx_params.n_batch = args.n_batch;
    ctx_params.n_ubatch = args.n_batch;
    ctx_params.n_seq_max = (args.n_parallel + 1) as u32;
    let ctx = Context(unsafe { ffi::llama_init_from_model(model.0, ctx_params) });
    if ctx.0.is_null() {
        return Err("failed to create llama_context".to_string());
    }

    let mem = unsafe { ffi::llama_get_memory(ctx.0) };
    let vocab = unsafe { ffi::llama_model_get_vocab(model.0) };
    if vocab.is_null() {
        return Err("failed to get model vocabulary".to_string());
    }

    let prompts = if args.prompt.is_empty() {
        eprintln!("No new questions so proceed with built-in defaults.");
        DEFAULT_PROMPTS
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    } else {
        eprintln!("Now printing the external prompt input");
        split_prompt_lines(&args.prompt)
    };

    let n_ctx = unsafe { ffi::llama_n_ctx(ctx.0) as i32 };
    let n_clients = args.n_parallel;
    let n_seq = args.n_sequences;
    let mut sampler_seed = args.seed;
    if sampler_seed >= 0 {
        eprintln!("initializing all samplers with the same RNG seed: {sampler_seed}");
    } else {
        eprintln!("initializing samplers with different RNG seeds, starting from {sampler_seed}");
    }

    let mut clients = Vec::new();
    for id in 0..n_clients {
        clients.push(Client {
            id,
            seq_id: -1,
            sampled: 0,
            t_start_prompt: 0,
            t_start_gen: 0,
            n_past: 0,
            n_prompt: 0,
            n_decoded: 0,
            i_batch: -1,
            input: String::new(),
            prompt: String::new(),
            response: String::new(),
            sampler: sampler_init(&args, sampler_seed),
        });
        if args.seed < 0 {
            sampler_seed -= 1;
        }
    }

    let tokens_system = tokenize(vocab, SYSTEM_PROMPT, true, false)?;
    let n_tokens_system = tokens_system.len() as i32;
    let mut batch = Batch(unsafe { ffi::llama_batch_init(n_ctx, 0, 1) });
    let mut n_total_prompt = 0;
    let mut n_total_gen = 0;
    let mut n_cache_miss = 0;
    let t_main_start = unsafe { ffi::ggml_time_us() };

    eprintln!(
        "Simulating parallel requests: n_parallel = {n_clients}, n_sequences = {n_seq}, cont_batching = {}, system tokens = {n_tokens_system}",
        args.cont_batching
    );

    if args.is_pp_shared {
        eprintln!("Evaluating the system prompt ...");
        for (i, token) in tokens_system.iter().enumerate() {
            unsafe {
                batch_add(&mut batch.0, *token, i as i32, &[0], false);
            }
        }
        if unsafe { ffi::llama_decode(ctx.0, batch.0) } != 0 {
            return Err("llama_decode failed for system prompt".to_string());
        }
        for i in 1..=n_clients {
            unsafe {
                ffi::llama_memory_seq_cp(mem, 0, i, -1, -1);
            }
        }
    }

    let mut rng = Lcg::new(1234);
    let mut g_seq_id = 0;
    eprintln!("Processing requests ...");

    loop {
        batch_clear(&mut batch.0);
        for client in &mut clients {
            if client.seq_id == -1 {
                continue;
            }
            client.i_batch = batch.0.n_tokens;
            unsafe {
                batch_add(
                    &mut batch.0,
                    client.sampled,
                    client.n_past,
                    &[client.id + 1],
                    true,
                );
            }
            client.n_past += 1;
            client.n_decoded += 1;
        }

        if batch.0.n_tokens == 0 {
            for i in 1..=n_clients {
                unsafe {
                    ffi::llama_memory_seq_rm(mem, i, -1, -1);
                    ffi::llama_memory_seq_cp(mem, 0, i, -1, -1);
                }
            }
            eprintln!("clearing the KV cache");
        }

        if args.cont_batching || batch.0.n_tokens == 0 {
            for client in &mut clients {
                if client.seq_id != -1 || g_seq_id >= n_seq {
                    continue;
                }
                client.seq_id = g_seq_id;
                client.t_start_prompt = unsafe { ffi::ggml_time_us() };
                client.t_start_gen = 0;
                client.input = prompts[rng.next_usize(prompts.len())].clone();
                client.response.clear();
                let (prompt, n_past, n_junk_cur) = build_prompt(
                    args.is_pp_shared,
                    n_tokens_system,
                    &client.input,
                    args.n_junk,
                    &mut rng,
                );
                client.prompt = prompt;
                client.n_past = n_past;
                unsafe {
                    ffi::llama_sampler_reset(client.sampler.0);
                }

                let tokens_prompt = tokenize(vocab, &client.prompt, false, false)?;
                for token in &tokens_prompt {
                    unsafe {
                        batch_add(&mut batch.0, *token, client.n_past, &[client.id + 1], false);
                    }
                    client.n_past += 1;
                }
                unsafe {
                    set_last_batch_logits(&mut batch.0);
                }
                client.n_prompt = tokens_prompt.len() as i32;
                client.n_decoded = 0;
                client.i_batch = batch.0.n_tokens - 1;
                eprintln!(
                    "Client {:3}, seq {:4}, junk = {:4}, prompt = {}, started decoding ...",
                    client.id, client.seq_id, n_junk_cur, client.n_prompt
                );
                g_seq_id += 1;
            }
        }

        if batch.0.n_tokens == 0 {
            break;
        }

        let mut n_batch = args.n_batch as i32;
        let mut i = 0;
        while i < batch.0.n_tokens {
            let n_tokens = n_batch.min(batch.0.n_tokens - i);
            let view = unsafe { batch_view(&batch.0, i, n_tokens) };
            let ret = unsafe { ffi::llama_decode(ctx.0, view) };
            if ret != 0 {
                if n_batch == 1 || ret < 0 {
                    return Err(format!(
                        "failed to decode the batch, n_batch = {n_batch}, ret = {ret}"
                    ));
                }
                eprintln!(
                    "failed to decode the batch, retrying with n_batch = {}",
                    n_batch / 2
                );
                n_cache_miss += 1;
                n_batch /= 2;
                continue;
            }

            let i_next = i + n_tokens;
            n_batch = args.n_batch as i32;
            for client in &mut clients {
                if client.i_batch < i || client.i_batch >= i_next {
                    continue;
                }
                let id = unsafe {
                    ffi::llama_sampler_sample(client.sampler.0, ctx.0, client.i_batch - i)
                };
                unsafe {
                    ffi::llama_sampler_accept(client.sampler.0, id);
                }
                if client.n_decoded == 1 {
                    client.t_start_gen = unsafe { ffi::ggml_time_us() };
                }
                let token_str = token_to_piece(vocab, id)?;
                client.response.push_str(&token_str);
                client.sampled = id;

                let stop_at_user = client.response.find("User:");
                let should_stop = client.n_decoded > 2
                    && (unsafe { ffi::llama_vocab_is_eog(vocab, id) }
                        || (args.n_predict > 0 && client.n_decoded >= args.n_predict)
                        || stop_at_user.is_some());
                if should_stop {
                    if let Some(pos) = stop_at_user {
                        client.response.truncate(pos);
                    }
                    unsafe {
                        ffi::llama_memory_seq_rm(mem, client.id + 1, -1, -1);
                        ffi::llama_memory_seq_cp(mem, 0, client.id + 1, -1, -1);
                    }
                    let t_main_end = unsafe { ffi::ggml_time_us() };
                    let seconds = (t_main_end - client.t_start_prompt) as f64 / 1e6;
                    let speed = (client.n_prompt + client.n_decoded) as f64 / seconds.max(1e-9);
                    eprintln!(
                        "Client {:3}, seq {:3}/{:3}, prompt {:4} t, response {:4} t, time {:5.2} s, speed {:5.2} t/s, cache miss {}\n\nInput:    {}\nResponse: {}\n",
                        client.id,
                        client.seq_id,
                        n_seq,
                        client.n_prompt,
                        client.n_decoded,
                        seconds,
                        speed,
                        n_cache_miss,
                        trim_ascii(&client.input),
                        trim_ascii(&client.response)
                    );
                    n_total_prompt += client.n_prompt;
                    n_total_gen += client.n_decoded;
                    client.seq_id = -1;
                }
                client.i_batch = -1;
            }
            i = i_next;
        }
    }

    let t_main_end = unsafe { ffi::ggml_time_us() };
    let seconds = (t_main_end - t_main_start) as f64 / 1e6;
    eprintln!(
        "n_parallel = {n_clients}, n_sequences = {n_seq}, cont_batching = {}, system tokens = {n_tokens_system}",
        args.cont_batching
    );
    eprintln!(
        "External prompt file: {}",
        if args.prompt_file.is_empty() {
            "used built-in defaults"
        } else {
            args.prompt_file.as_str()
        }
    );
    eprintln!("Model and path used:  {}\n", args.model_path);
    eprintln!(
        "Total prompt tokens: {n_total_prompt:6}, speed: {:5.2} t/s",
        n_total_prompt as f64 / seconds.max(1e-9)
    );
    eprintln!(
        "Total gen tokens:    {n_total_gen:6}, speed: {:5.2} t/s",
        n_total_gen as f64 / seconds.max(1e-9)
    );
    eprintln!(
        "Total speed (AVG):           speed: {:5.2} t/s",
        (n_total_prompt + n_total_gen) as f64 / seconds.max(1e-9)
    );
    eprintln!("Cache misses:        {n_cache_miss:6}\n");
    unsafe {
        ffi::llama_perf_context_print(ctx.0);
    }
    Ok(())
}

fn print_usage(program: &str) {
    eprintln!();
    eprintln!("example usage:");
    eprintln!();
    eprintln!("    {program} -m model.gguf -np 4 -ns 8");
    eprintln!();
}

fn main() {
    let mut argv = env::args();
    let program = argv.next().unwrap_or_else(|| "llama-parallel".to_string());
    let args = match parse_args(argv) {
        Ok(args) => args,
        Err(err) => {
            eprintln!("{program}: error: {err}");
            print_usage(&program);
            std::process::exit(1);
        }
    };
    if let Err(err) = run(args) {
        eprintln!("{program}: error: {err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_parallel_options() {
        let args = parse_args([
            "-m",
            "model.gguf",
            "-p",
            "q1\nq2",
            "-n",
            "16",
            "-np",
            "3",
            "-ns",
            "7",
            "--no-cont-batching",
            "-pps",
            "--junk",
            "4",
            "-c",
            "1024",
            "-b",
            "128",
            "-ngl",
            "0",
            "--top-k",
            "10",
            "--top-p",
            "0.8",
            "--temp",
            "0.5",
            "-s",
            "-2",
        ])
        .unwrap();
        assert_eq!(args.model_path, "model.gguf");
        assert_eq!(args.prompt, "q1\nq2");
        assert_eq!(args.n_predict, 16);
        assert_eq!(args.n_parallel, 3);
        assert_eq!(args.n_sequences, 7);
        assert!(!args.cont_batching);
        assert!(args.is_pp_shared);
        assert_eq!(args.n_junk, 4);
        assert_eq!(args.n_ctx, 1024);
        assert_eq!(args.n_batch, 128);
        assert_eq!(args.n_gpu_layers, 0);
        assert_eq!(args.top_k, 10);
        assert_eq!(args.top_p, 0.8);
        assert_eq!(args.temp, 0.5);
        assert_eq!(args.seed, -2);
    }

    #[test]
    fn trims_ascii_whitespace() {
        assert_eq!(trim_ascii(" \nhello\t"), "hello");
    }

    #[test]
    fn splits_prompt_lines() {
        assert_eq!(split_prompt_lines("a\nb\n"), vec!["a", "b"]);
    }

    #[test]
    fn deterministic_rng_repeats() {
        let mut a = Lcg::new(1234);
        let mut b = Lcg::new(1234);
        assert_eq!(a.next_usize(9), b.next_usize(9));
        assert_eq!(a.next_usize(20), b.next_usize(20));
    }

    #[test]
    fn builds_prompt_with_shared_system() {
        let mut rng = Lcg::new(1234);
        let (prompt, n_past, n_junk) = build_prompt(true, 5, "Question?", 1, &mut rng);
        assert_eq!(n_past, 5);
        assert_eq!(n_junk, 0);
        assert_eq!(prompt, "User:\nQuestion?\nAssistant:\n");
    }
}
