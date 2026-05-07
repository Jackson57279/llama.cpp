use llama_simple_chat_rust::{parse_args, Args};
use llama_simple_rust::ffi;
use std::env;
use std::ffi::{c_char, c_void, CString};
use std::io::{self, Write};
use std::ptr;

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

unsafe extern "C" fn log_errors_only(
    level: ffi::ggml_log_level,
    text: *const c_char,
    _: *mut c_void,
) {
    if level >= 2 && !text.is_null() {
        let mut len = 0;
        while *text.add(len) != 0 {
            len += 1;
        }
        let bytes = std::slice::from_raw_parts(text as *const u8, len);
        eprint!("{}", String::from_utf8_lossy(bytes));
    }
}

fn print_usage(program: &str) {
    eprintln!();
    eprintln!("example usage:");
    eprintln!();
    eprintln!("    {program} -m model.gguf [-c context_size] [-ngl n_gpu_layers]");
    eprintln!();
}

fn token_to_piece(
    vocab: *const ffi::llama_vocab,
    token: ffi::llama_token,
) -> Result<String, String> {
    let mut buf = vec![0_i8; 256];
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

fn generate(
    ctx: *mut ffi::llama_context,
    vocab: *const ffi::llama_vocab,
    sampler: *mut ffi::llama_sampler,
    prompt: &str,
) -> Result<String, String> {
    let c_prompt = CString::new(prompt).map_err(|_| "prompt contains an interior NUL byte")?;
    let memory = unsafe { ffi::llama_get_memory(ctx) };
    let is_first = unsafe { ffi::llama_memory_seq_pos_max(memory, 0) } == -1;

    let n_prompt_tokens = unsafe {
        -ffi::llama_tokenize(
            vocab,
            c_prompt.as_ptr(),
            prompt.len() as i32,
            ptr::null_mut(),
            0,
            is_first,
            true,
        )
    };
    if n_prompt_tokens <= 0 {
        return Err("failed to size prompt tokenization".to_string());
    }

    let mut prompt_tokens = vec![0_i32; n_prompt_tokens as usize];
    let n_tokenized = unsafe {
        ffi::llama_tokenize(
            vocab,
            c_prompt.as_ptr(),
            prompt.len() as i32,
            prompt_tokens.as_mut_ptr(),
            prompt_tokens.len() as i32,
            is_first,
            true,
        )
    };
    if n_tokenized < 0 {
        return Err("failed to tokenize the prompt".to_string());
    }

    let mut batch =
        unsafe { ffi::llama_batch_get_one(prompt_tokens.as_mut_ptr(), prompt_tokens.len() as i32) };
    let mut response = String::new();
    let mut token_storage = [0_i32; 1];

    loop {
        let n_ctx = unsafe { ffi::llama_n_ctx(ctx) as i32 };
        let n_ctx_used = unsafe { ffi::llama_memory_seq_pos_max(memory, 0) } + 1;
        if n_ctx_used + batch.n_tokens > n_ctx {
            println!("\x1b[0m");
            return Err("context size exceeded".to_string());
        }

        let ret = unsafe { ffi::llama_decode(ctx, batch) };
        if ret != 0 {
            return Err(format!("failed to decode, ret = {ret}"));
        }

        token_storage[0] = unsafe { ffi::llama_sampler_sample(sampler, ctx, -1) };
        let new_token_id = token_storage[0];
        if unsafe { ffi::llama_vocab_is_eog(vocab, new_token_id) } {
            break;
        }

        let piece = token_to_piece(vocab, new_token_id)?;
        print!("{piece}");
        io::stdout().flush().map_err(|err| err.to_string())?;
        response.push_str(&piece);

        batch = unsafe { ffi::llama_batch_get_one(token_storage.as_mut_ptr(), 1) };
    }

    Ok(response)
}

fn run(args: Args) -> Result<(), String> {
    let model_path = CString::new(args.model_path.as_str())
        .map_err(|_| "model path contains an interior NUL byte".to_string())?;

    unsafe {
        ffi::llama_log_set(Some(log_errors_only), ptr::null_mut());
        ffi::ggml_backend_load_all();
    }

    let mut model_params = unsafe { ffi::llama_model_default_params() };
    model_params.n_gpu_layers = args.n_gpu_layers;
    let model =
        Model(unsafe { ffi::llama_model_load_from_file(model_path.as_ptr(), model_params) });
    if model.0.is_null() {
        return Err("unable to load model".to_string());
    }

    let vocab = unsafe { ffi::llama_model_get_vocab(model.0) };
    if vocab.is_null() {
        return Err("failed to get model vocabulary".to_string());
    }

    let mut ctx_params = unsafe { ffi::llama_context_default_params() };
    ctx_params.n_ctx = args.n_ctx as u32;
    ctx_params.n_batch = args.n_ctx as u32;
    let ctx = Context(unsafe { ffi::llama_init_from_model(model.0, ctx_params) });
    if ctx.0.is_null() {
        return Err("failed to create the llama_context".to_string());
    }

    let sampler = Sampler(unsafe {
        ffi::llama_sampler_chain_init(ffi::llama_sampler_chain_default_params())
    });
    if sampler.0.is_null() {
        return Err("failed to create sampler chain".to_string());
    }

    unsafe {
        ffi::llama_sampler_chain_add(sampler.0, ffi::llama_sampler_init_min_p(0.05, 1));
        ffi::llama_sampler_chain_add(sampler.0, ffi::llama_sampler_init_temp(0.8));
        ffi::llama_sampler_chain_add(
            sampler.0,
            ffi::llama_sampler_init_dist(ffi::LLAMA_DEFAULT_SEED),
        );
    }

    let mut messages = Vec::<(CString, CString)>::new();
    let mut formatted = vec![0_i8; unsafe { ffi::llama_n_ctx(ctx.0) } as usize];
    let mut prev_len = 0_i32;

    loop {
        print!("\x1b[32m> \x1b[0m");
        io::stdout().flush().map_err(|err| err.to_string())?;

        let mut user = String::new();
        io::stdin()
            .read_line(&mut user)
            .map_err(|err| err.to_string())?;
        let user = user.trim_end_matches(['\r', '\n']).to_string();
        if user.is_empty() {
            break;
        }

        let role_user = CString::new("user").unwrap();
        let content_user =
            CString::new(user).map_err(|_| "user input contains an interior NUL byte")?;
        messages.push((role_user, content_user));

        let c_messages = messages
            .iter()
            .map(|(role, content)| ffi::llama_chat_message {
                role: role.as_ptr(),
                content: content.as_ptr(),
            })
            .collect::<Vec<_>>();
        let tmpl = unsafe { ffi::llama_model_chat_template(model.0, ptr::null()) };
        let mut new_len = unsafe {
            ffi::llama_chat_apply_template(
                tmpl,
                c_messages.as_ptr(),
                c_messages.len(),
                true,
                formatted.as_mut_ptr(),
                formatted.len() as i32,
            )
        };
        if new_len > formatted.len() as i32 {
            formatted.resize(new_len as usize, 0);
            new_len = unsafe {
                ffi::llama_chat_apply_template(
                    tmpl,
                    c_messages.as_ptr(),
                    c_messages.len(),
                    true,
                    formatted.as_mut_ptr(),
                    formatted.len() as i32,
                )
            };
        }
        if new_len < 0 {
            return Err("failed to apply the chat template".to_string());
        }

        let prompt_bytes = formatted[prev_len as usize..new_len as usize]
            .iter()
            .map(|&c| c as u8)
            .collect::<Vec<_>>();
        let prompt = String::from_utf8_lossy(&prompt_bytes).into_owned();

        print!("\x1b[33m");
        let response = generate(ctx.0, vocab, sampler.0, &prompt)?;
        println!("\n\x1b[0m");

        let role_assistant = CString::new("assistant").unwrap();
        let content_assistant =
            CString::new(response).map_err(|_| "response contains an interior NUL byte")?;
        messages.push((role_assistant, content_assistant));

        let c_messages = messages
            .iter()
            .map(|(role, content)| ffi::llama_chat_message {
                role: role.as_ptr(),
                content: content.as_ptr(),
            })
            .collect::<Vec<_>>();
        prev_len = unsafe {
            ffi::llama_chat_apply_template(
                tmpl,
                c_messages.as_ptr(),
                c_messages.len(),
                false,
                ptr::null_mut(),
                0,
            )
        };
        if prev_len < 0 {
            return Err("failed to apply the chat template".to_string());
        }
    }

    Ok(())
}

fn main() {
    let mut argv = env::args();
    let program = argv
        .next()
        .unwrap_or_else(|| "llama-simple-chat".to_string());

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
