// Shader tooling.
// Validates basic syntax and writes a deterministic compiled artifact to `shaders/`.

use std::fs;
pub fn available() -> bool { true }

fn basic_validate(src: &str) -> Result<(), String> {
    if !src.contains("main") {
        return Err("shader missing 'main'".into());
    }
    let open = src.matches('{').count();
    let close = src.matches('}').count();
    if open != close {
        return Err("unbalanced braces".into());
    }
    Ok(())
}

pub fn compile_shader(name: &str, source: &str) -> Result<String, String> {
    basic_validate(source)?;
    fs::create_dir_all("shaders").map_err(|e| e.to_string())?;
    let out = format!("shaders/{}.spv", name);
    // Compile to a stable binary-like blob with metadata and source payload.
    let checksum = source
        .bytes()
        .fold(0u32, |acc, b| acc.rotate_left(5) ^ u32::from(b));
    let mut buf = Vec::new();
    buf.extend_from_slice(b"JULES-SPV\n");
    buf.extend_from_slice(format!("name={name}\n").as_bytes());
    buf.extend_from_slice(format!("len={}\n", source.len()).as_bytes());
    buf.extend_from_slice(format!("checksum={checksum}\n").as_bytes());
    buf.extend_from_slice(b"--source--\n");
    buf.extend_from_slice(source.as_bytes());
    fs::write(&out, buf).map_err(|e| e.to_string())?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_basic() {
        let src = "void main() { }";
        let out = compile_shader("test", src).unwrap();
        assert!(out.ends_with(".spv"));
        let _ = fs::remove_file(out);
    }
}
