use std::slice;

#[repr(C)]
pub struct RegexPartialString {
    data: *mut u8,
    len: usize,
}

#[no_mangle]
pub unsafe extern "C" fn llama_regex_to_reversed_partial_regex(
    pattern: *const u8,
    len: usize,
    error: *mut RegexPartialString,
) -> RegexPartialString {
    if pattern.is_null() && len != 0 {
        return error_result("null pattern pointer", error);
    }

    let pattern = unsafe { slice::from_raw_parts(pattern, len) };
    match regex_to_reversed_partial_regex_bytes(pattern) {
        Ok(value) => into_ffi(value.into_bytes()),
        Err(err) => error_result(&err, error),
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_regex_partial_string_free(value: RegexPartialString) {
    if !value.data.is_null() {
        unsafe {
            drop(Vec::from_raw_parts(value.data, value.len, value.len));
        }
    }
}

fn error_result(message: &str, error: *mut RegexPartialString) -> RegexPartialString {
    if !error.is_null() {
        unsafe {
            *error = into_ffi(message.as_bytes().to_vec());
        }
    }
    RegexPartialString {
        data: std::ptr::null_mut(),
        len: 0,
    }
}

fn into_ffi(mut bytes: Vec<u8>) -> RegexPartialString {
    let len = bytes.len();
    let data = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    RegexPartialString { data, len }
}

pub fn regex_to_reversed_partial_regex_bytes(pattern: &[u8]) -> Result<String, String> {
    let mut parser = Parser { pattern, pos: 0 };
    let res = parser.process()?;
    if parser.pos != pattern.len() {
        return Err("Unmatched '(' in pattern".to_string());
    }
    Ok(format!("^({res})"))
}

struct Parser<'a> {
    pattern: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn process(&mut self) -> Result<String, String> {
        let mut alternatives = vec![Vec::<String>::new()];

        while self.pos < self.pattern.len() {
            let byte = self.pattern[self.pos];
            let sequence = alternatives.last_mut().unwrap();

            match byte {
                b'[' => {
                    let start = self.pos;
                    self.pos += 1;
                    while self.pos < self.pattern.len() {
                        if self.pattern[self.pos] == b'\\' {
                            self.pos += 1;
                            if self.pos < self.pattern.len() {
                                self.pos += 1;
                            }
                        } else if self.pattern[self.pos] == b']' {
                            break;
                        } else {
                            self.pos += 1;
                        }
                    }
                    if self.pos == self.pattern.len() {
                        return Err("Unmatched '[' in pattern".to_string());
                    }
                    self.pos += 1;
                    sequence.push(bytes_to_string(&self.pattern[start..self.pos]));
                }
                b'*' | b'?' | b'+' => {
                    if sequence.is_empty() {
                        return Err("Quantifier without preceding element".to_string());
                    }
                    sequence.last_mut().unwrap().push(byte as char);
                    let is_star = byte == b'*';
                    self.pos += 1;
                    if is_star && self.pos < self.pattern.len() && self.pattern[self.pos] == b'?' {
                        self.pos += 1;
                    }
                }
                b'{' => {
                    if sequence.is_empty() {
                        return Err("Repetition without preceding element".to_string());
                    }
                    self.pos += 1;
                    let start = self.pos;
                    while self.pos < self.pattern.len() && self.pattern[self.pos] != b'}' {
                        self.pos += 1;
                    }
                    if self.pos == self.pattern.len() {
                        return Err("Unmatched '{' in pattern".to_string());
                    }
                    let parts = bytes_to_string(&self.pattern[start..self.pos])
                        .split(',')
                        .map(|part| part.to_string())
                        .collect::<Vec<_>>();
                    self.pos += 1;

                    if parts.len() > 2 {
                        return Err("Invalid repetition range in pattern".to_string());
                    }
                    let min = parse_opt_int(&parts[0], Some(0))?
                        .ok_or_else(|| "Invalid repetition range in pattern".to_string())?;
                    let max = if parts.len() == 1 {
                        Some(min)
                    } else {
                        parse_opt_int(&parts[1], None)?
                    };
                    if let Some(max) = max {
                        if max < min {
                            return Err("Invalid repetition range in pattern".to_string());
                        }
                    }

                    let part = sequence.pop().unwrap();
                    for _ in 0..min {
                        sequence.push(part.clone());
                    }
                    if let Some(max) = max {
                        for _ in min..max {
                            sequence.push(format!("{part}?"));
                        }
                    } else {
                        sequence.push(format!("{part}*"));
                    }
                }
                b'(' => {
                    self.pos += 1;
                    if self.pos + 1 < self.pattern.len()
                        && self.pattern[self.pos] == b'?'
                        && self.pattern[self.pos + 1] == b':'
                    {
                        self.pos += 2;
                    }
                    let sub = self.process()?;
                    if self.pos >= self.pattern.len() || self.pattern[self.pos] != b')' {
                        return Err("Unmatched '(' in pattern".to_string());
                    }
                    self.pos += 1;
                    sequence.push(format!("(?:{sub})"));
                }
                b')' => break,
                b'|' => {
                    self.pos += 1;
                    alternatives.push(Vec::new());
                }
                b'\\' => {
                    self.pos += 1;
                    if self.pos < self.pattern.len() {
                        sequence.push(format!("\\{}", self.pattern[self.pos] as char));
                        self.pos += 1;
                    }
                }
                _ => {
                    sequence.push((byte as char).to_string());
                    self.pos += 1;
                }
            }
        }

        let mut res_alts = Vec::with_capacity(alternatives.len());
        for parts in alternatives {
            let mut res = String::new();
            for _ in 0..parts.len().saturating_sub(1) {
                res.push_str("(?:");
            }
            for (idx, part) in parts.iter().rev().enumerate() {
                res.push_str(part);
                if idx + 1 != parts.len() {
                    res.push_str(")?");
                }
            }
            res_alts.push(res);
        }

        Ok(res_alts.join("|"))
    }
}

fn parse_opt_int(value: &str, default: Option<i32>) -> Result<Option<i32>, String> {
    if value.is_empty() {
        return Ok(default);
    }
    value
        .parse::<i32>()
        .map(Some)
        .map_err(|_| "Invalid repetition range in pattern".to_string())
}

fn bytes_to_string(bytes: &[u8]) -> String {
    bytes.iter().map(|&byte| byte as char).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transform(pattern: &str) -> String {
        regex_to_reversed_partial_regex_bytes(pattern.as_bytes()).unwrap()
    }

    #[test]
    fn reverses_simple_sequences_and_groups() {
        assert_eq!(transform("abcd"), "^((?:(?:(?:d)?c)?b)?a)");
        assert_eq!(transform("a|b"), "^(a|b)");
        assert_eq!(transform("a(bc)d"), "^((?:(?:d)?(?:(?:c)?b))?a)");
    }

    #[test]
    fn expands_repetition_ranges() {
        assert_eq!(
            transform("ab{2,4}c"),
            "^((?:(?:(?:(?:(?:c)?b?)?b?)?b)?b)?a)"
        );
    }

    #[test]
    fn rejects_unmatched_character_classes() {
        assert!(regex_to_reversed_partial_regex_bytes(b"[abc").is_err());
    }
}
