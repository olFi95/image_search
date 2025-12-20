use std::path::PathBuf;
use std::process::Command;

pub fn init() -> Result<(), anyhow::Error> {
    // Step 2: Set up Python venv under ./target/venv
    let venv_dir = get_venv_dir()?;
    let pip_path = if cfg!(target_os = "windows") {
        venv_dir.join("Scripts/pip.exe")
    } else {
        venv_dir.join("bin/pip")
    };

    println!("cargo:warning=pip path {}", pip_path.clone().into_os_string().into_string().unwrap());
    println!("cargo:warning=Installing Python dependencies...");
    let status = Command::new(&pip_path)
        .arg("install")
        .arg("--upgrade")
        .arg("pip")
        .status()?;
    if !status.success() {
        panic!("Failed to upgrade pip");
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let requirements = manifest_dir
        .join("python")
        .join("requirements.txt");
    let status = Command::new(&pip_path)
        .arg("install")
        .arg("-r")
        .arg(&requirements)
        .status()?;    if !status.success() {
        panic!("Failed to install required Python packages");
    }

    Ok(())
}

pub fn get_venv_dir() -> Result<PathBuf, anyhow::Error> {
    let venv_dir = PathBuf::from("..").join(".venv");
    if !venv_dir.exists() {
        println!("cargo:warning=Creating Python venv...");
        let python = which::which("python3")?;
        let status = Command::new(python)
            .arg("-m")
            .arg("venv")
            .arg(&venv_dir)
            .status()?;
        if !status.success() {
            panic!("Failed to create Python virtual environment");
        }
    }
    Ok(venv_dir)
}

pub fn update(input_model: &PathBuf, output_model: &PathBuf) -> Result<(), anyhow::Error> {
    let venv_dir = get_venv_dir()?;
    let python_path = if cfg!(target_os = "windows") {
        venv_dir.join("Scripts/python.exe")
    } else {
        venv_dir.join("bin/python3")
    };
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let upgrade_script = manifest_dir
        .join("python")
        .join("upgrade_opset.py");

    println!("Running opset upgrade script...");
    let status = Command::new(&python_path)
        .arg(&upgrade_script)
        .arg(input_model.as_os_str())
        .arg(output_model.as_os_str())
        .status()?;

    if !status.success() {
        panic!("Opset upgrade script to upgrade ONNX model failed.");
    }

    Ok(())
}
