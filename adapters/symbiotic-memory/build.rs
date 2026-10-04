use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

const SYMBIOTIC_MEMORY_REV: &str = "b99ff63f379993c6a1c2cbef611a8a22d26b7056";

fn main() {
    println!("cargo:rerun-if-env-changed=ZVEC_LIB_DIR");
    println!("cargo:rerun-if-env-changed=ZVEC_LIB_SHA256");
    let dir = zvec_lib_dir().unwrap_or_else(|| {
        panic!(
            "no zvec library proven from locked Symbiotic Memory revision {SYMBIOTIC_MEMORY_REV}; \
             fetch the locked adapter dependencies or set ZVEC_LIB_DIR plus ZVEC_LIB_SHA256"
        )
    });
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{}", dir.display());
    println!("cargo:rustc-env=MEMBENCH_ZVEC_SOURCE_REV={SYMBIOTIC_MEMORY_REV}");

    let sha = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=GIT_SHA={sha}");
}

fn zvec_lib_dir() -> Option<PathBuf> {
    let vendored = |root: &Path| root.join("vendor/zvec-rust/vendor/lib");
    if let Ok(dir) = std::env::var("ZVEC_LIB_DIR") {
        let dir = PathBuf::from(dir);
        verify_explicit_lib_dir(&dir);
        return Some(dir);
    }

    let cargo_home = std::env::var("CARGO_HOME")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|home| PathBuf::from(home).join(".cargo"))
        })?;
    let mut candidates = Vec::new();
    for repo in std::fs::read_dir(cargo_home.join("git/checkouts"))
        .ok()?
        .flatten()
    {
        if !repo
            .file_name()
            .to_string_lossy()
            .starts_with("symbiotic-memory-")
        {
            continue;
        }
        for revision in std::fs::read_dir(repo.path()).ok()?.flatten() {
            if checkout_revision(&revision.path()).as_deref() != Some(SYMBIOTIC_MEMORY_REV) {
                continue;
            }
            let candidate = vendored(&revision.path());
            if shared_library(&candidate).is_some() {
                candidates.push(candidate);
            }
        }
    }
    candidates.sort();
    candidates.dedup();
    candidates.into_iter().next()
}

fn shared_library(dir: &Path) -> Option<PathBuf> {
    let name = match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => "libzvec_c_api.dylib",
        Ok("linux") => "libzvec_c_api.so",
        _ => return None,
    };
    let path = dir.join(name);
    path.is_file().then_some(path)
}

fn checkout_revision(root: &Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["-C", root.to_str()?, "rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
}

fn verify_explicit_lib_dir(dir: &Path) {
    let library = shared_library(dir).unwrap_or_else(|| {
        panic!(
            "ZVEC_LIB_DIR contains no supported zvec shared library: {}",
            dir.display()
        )
    });
    let expected = std::env::var("ZVEC_LIB_SHA256")
        .unwrap_or_else(|_| panic!("ZVEC_LIB_SHA256 is required whenever ZVEC_LIB_DIR is set"));
    assert!(
        expected.len() == 64 && expected.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "ZVEC_LIB_SHA256 must be one 64-character hexadecimal digest"
    );
    let mut file = std::fs::File::open(&library)
        .unwrap_or_else(|error| panic!("cannot open {}: {error}", library.display()));
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .unwrap_or_else(|error| panic!("cannot hash {}: {error}", library.display()));
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let actual = format!("{:x}", hasher.finalize());
    assert_eq!(
        actual,
        expected.to_ascii_lowercase(),
        "ZVEC_LIB_SHA256 does not match {}",
        library.display()
    );
}
