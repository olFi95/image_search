// =============================================================================
// Image Downloader
// Downloads images from a TSV file containing URLs.
// Rust port of test_pictures/downloader.sh
// =============================================================================

use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use anyhow::Context;
use clap::Parser;
use governor::{DefaultDirectRateLimiter, Quota, RateLimiter};
use reqwest::StatusCode;
use std::num::NonZeroU32;
use tokio::sync::Semaphore;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TSV_URL: &str =
    "https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-test.tsv";
const DEFAULT_TSV_FILENAME: &str = "open-images-dataset-test.tsv";

const MAX_RETRIES: u32 = 3;
const RETRY_DELAY_SECS: u64 = 2;
const TIMEOUT_SECS: u64 = 15;
const DEFAULT_PARALLEL_JOBS: usize = 5;

// ---------------------------------------------------------------------------
// ANSI colours
// ---------------------------------------------------------------------------

const RED: &str = "\x1b[0;31m";
const GREEN: &str = "\x1b[0;32m";
const YELLOW: &str = "\x1b[1;33m";
const BLUE: &str = "\x1b[0;34m";
const NC: &str = "\x1b[0m"; // No Color

macro_rules! log_info    { ($($arg:tt)*) => { println!("{BLUE}[INFO]{NC} {}", format_args!($($arg)*)) } }
macro_rules! log_success { ($($arg:tt)*) => { println!("{GREEN}[SUCCESS]{NC} {}", format_args!($($arg)*)) } }
macro_rules! log_warning { ($($arg:tt)*) => { println!("{YELLOW}[WARNING]{NC} {}", format_args!($($arg)*)) } }
macro_rules! log_error   { ($($arg:tt)*) => { eprintln!("{RED}[ERROR]{NC} {}", format_args!($($arg)*)) } }

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Downloads images from the Open Images Dataset TSV file.
///
/// Usage examples:
///   downloader ./images 100
///   downloader ./images 100 10
///   downloader --skip 500 ./images 100 10
#[derive(Parser, Debug)]
#[command(verbatim_doc_comment)]
struct Args {
    /// Destination folder for downloaded images.
    output_folder: PathBuf,

    /// Maximum number of images to download.
    num_images: usize,

    /// Number of parallel downloads (default: 5).
    #[arg(default_value_t = DEFAULT_PARALLEL_JOBS)]
    parallel_jobs: usize,

    /// Skip the first N valid image URLs in the TSV file.
    #[arg(long, default_value_t = 0)]
    skip: usize,

    /// Path to the TSV file; downloaded automatically when absent.
    /// Defaults to `open-images-dataset-test.tsv` in the current directory.
    #[arg(long)]
    tsv_file: Option<PathBuf>,

    /// Maximum HTTP requests per hour across all parallel workers.
    /// Flickr's hard cap is 3600/hour; default stays safely under at 3000/hour. This applies to requests with API token though.
    /// https://www.flickr.com/services/developer/api/
    #[arg(long, default_value_t = 3000)]
    rate_limit: u32,
}

// ---------------------------------------------------------------------------
// Download result
// ---------------------------------------------------------------------------

enum DownloadResult {
    Success,
    Skipped,
    Failed,
}

// ---------------------------------------------------------------------------
// Per-run counters
// ---------------------------------------------------------------------------

struct Counters {
    success: AtomicU64,
    failed: AtomicU64,
    skipped: AtomicU64,
}

impl Counters {
    fn new() -> Self {
        Self {
            success: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
        }
    }
}

// ---------------------------------------------------------------------------
// TSV download
// ---------------------------------------------------------------------------

async fn download_tsv_if_needed(tsv_path: &Path, client: &reqwest::Client) -> anyhow::Result<()> {
    if tsv_path.exists() {
        let meta = std::fs::metadata(tsv_path)?;
        if meta.len() > 0 {
            log_info!("TSV file already exists: {}", tsv_path.display());
            return Ok(());
        }
    }

    log_info!("Downloading Open Images Dataset TSV file...");
    log_info!("Source: {TSV_URL}");

    let response = client
        .get(TSV_URL)
        .send()
        .await
        .context("Failed to send TSV download request")?;

    if !response.status().is_success() {
        anyhow::bail!(
            "Failed to download TSV file: HTTP {}",
            response.status()
        );
    }

    let bytes = response
        .bytes()
        .await
        .context("Failed to read TSV response body")?;

    tokio::fs::write(tsv_path, &bytes)
        .await
        .with_context(|| format!("Failed to write TSV to {}", tsv_path.display()))?;

    log_success!("TSV file downloaded successfully");
    Ok(())
}

// ---------------------------------------------------------------------------
// Single image download
// ---------------------------------------------------------------------------

async fn download_image(
    client: &reqwest::Client,
    url: &str,
    output_folder: &Path,
    index: usize,
    limiter: &DefaultDirectRateLimiter,
) -> DownloadResult {
    // Derive filename from URL (strip query string)
    let filename = url
        .split('?')
        .next()
        .and_then(|u| u.rsplit('/').next())
        .filter(|f| !f.is_empty() && *f != "/")
        .map(|f| f.to_string())
        .unwrap_or_else(|| format!("image_{index}.jpg"));

    let file_path = output_folder.join(&filename);

    // Skip already-downloaded files
    if file_path.exists() {
        if file_path.metadata().map(|m| m.len() > 0).unwrap_or(false) {
            log_warning!("File already exists, skipping: {filename}");
            return DownloadResult::Skipped;
        }
        // Zero-byte leftover from a previous run – try again
        let _ = std::fs::remove_file(&file_path);
    }

    let mut attempt = 0u32;
    loop {
        attempt += 1;

        // Acquire a rate-limit token before every HTTP request (including retries).
        limiter.until_ready().await;

        let response = match client.get(url).send().await {
            Ok(r) => r,
            Err(e) => {
                log_warning!("Request error (attempt {attempt}/{MAX_RETRIES}): {e}");
                if attempt < MAX_RETRIES {
                    tokio::time::sleep(tokio::time::Duration::from_secs(RETRY_DELAY_SECS)).await;
                    continue;
                }
                log_error!("Failed after {MAX_RETRIES} attempts: {url}");
                return DownloadResult::Failed;
            }
        };

        let status = response.status();

        // --- HTTP status handling ---
        if status == StatusCode::NOT_FOUND {
            // 404 – resource is gone, don't retry
            log_error!("404 Not Found (no retry): {url}");
            return DownloadResult::Failed;
        }

        if status == StatusCode::GONE {
            // 410 – resource is permanently gone, don't retry
            log_error!("410 Gone (no retry): {url}");
            return DownloadResult::Failed;
        }

        if status == StatusCode::TOO_MANY_REQUESTS {
            // 429 – back off and retry
            log_warning!("429 Too Many Requests (attempt {attempt}/{MAX_RETRIES}): {url}");
            if attempt < MAX_RETRIES {
                tokio::time::sleep(tokio::time::Duration::from_secs(RETRY_DELAY_SECS)).await;
                continue;
            }
            log_error!("Failed after {MAX_RETRIES} attempts (rate-limited): {url}");
            return DownloadResult::Failed;
        }

        if !status.is_success() {
            log_warning!("HTTP {status} (attempt {attempt}/{MAX_RETRIES}): {url}");
            if attempt < MAX_RETRIES {
                tokio::time::sleep(tokio::time::Duration::from_secs(RETRY_DELAY_SECS)).await;
                continue;
            }
            log_error!("Failed after {MAX_RETRIES} attempts (HTTP {status}): {url}");
            return DownloadResult::Failed;
        }

        // --- Read response body ---
        let bytes = match response.bytes().await {
            Ok(b) if !b.is_empty() => b,
            Ok(_) => {
                log_warning!("Empty response body (attempt {attempt}/{MAX_RETRIES}): {url}");
                if attempt < MAX_RETRIES {
                    tokio::time::sleep(tokio::time::Duration::from_secs(RETRY_DELAY_SECS)).await;
                    continue;
                }
                log_error!("Failed after {MAX_RETRIES} attempts (empty body): {url}");
                return DownloadResult::Failed;
            }
            Err(e) => {
                log_warning!("Failed to read body (attempt {attempt}/{MAX_RETRIES}): {e}");
                if attempt < MAX_RETRIES {
                    tokio::time::sleep(tokio::time::Duration::from_secs(RETRY_DELAY_SECS)).await;
                    continue;
                }
                log_error!("Failed after {MAX_RETRIES} attempts: {url}");
                return DownloadResult::Failed;
            }
        };

        // --- Validate: try to decode as an image ---
        if let Err(e) = image::load_from_memory(&bytes) {
            log_warning!("Invalid or corrupt image ({e}) (attempt {attempt}/{MAX_RETRIES}): {filename}");
            if attempt < MAX_RETRIES {
                // The file is likely the same on retry, but we mirror the bash behaviour
                tokio::time::sleep(tokio::time::Duration::from_secs(RETRY_DELAY_SECS)).await;
                continue;
            }
            log_error!("Giving up on corrupt image after {MAX_RETRIES} attempts: {filename}");
            return DownloadResult::Failed;
        }

        // --- Write to disk (via temp file to avoid partial writes) ---
        let tmp_path = file_path.with_extension("tmp");
        if let Err(e) = tokio::fs::write(&tmp_path, &bytes).await {
            log_error!("Failed to write temporary file for {filename}: {e}");
            return DownloadResult::Failed;
        }
        if let Err(e) = tokio::fs::rename(&tmp_path, &file_path).await {
            log_error!("Failed to rename temporary file for {filename}: {e}");
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return DownloadResult::Failed;
        }

        log_success!("Downloaded: {filename}");
        return DownloadResult::Success;
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.num_images == 0 {
        anyhow::bail!("num_images must be a positive integer");
    }

    // HTTP client shared across all tasks
    let client = Arc::new(
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(TIMEOUT_SECS))
            .build()
            .context("Failed to build HTTP client")?,
    );

    // Resolve TSV path – default: <cwd>/open-images-dataset-test.tsv
    let tsv_path = args.tsv_file.unwrap_or_else(|| {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(DEFAULT_TSV_FILENAME)
    });

    download_tsv_if_needed(&tsv_path, &client).await?;

    // Create output folder
    tokio::fs::create_dir_all(&args.output_folder)
        .await
        .with_context(|| format!("Failed to create output folder: {}", args.output_folder.display()))?;

    log_info!("Starting download of up to {} image(s)", args.num_images);
    log_info!("Source TSV : {}", tsv_path.display());
    log_info!("Destination: {}", args.output_folder.display());
    log_info!("Parallel   : {}", args.parallel_jobs);
    log_info!(
        "Rate limit : {}/hour  ({:.1} req/s, Flickr cap is 3600/hour)",
        args.rate_limit,
        args.rate_limit as f64 / 3600.0
    );
    if args.skip > 0 {
        log_info!("Skipping first {} valid URL(s) in TSV", args.skip);
    }
    println!();

    // Read and filter URLs from TSV
    let tsv_content = tokio::fs::read_to_string(&tsv_path)
        .await
        .with_context(|| format!("Failed to read TSV file: {}", tsv_path.display()))?;

    let urls: Vec<String> = tsv_content
        .lines()
        .skip(1) // skip header row
        .filter_map(|line| {
            let url = line.split('\t').next()?;
            if url.starts_with("http://") || url.starts_with("https://") {
                Some(url.to_string())
            } else {
                None
            }
        })
        .skip(args.skip)
        .take(args.num_images)
        .collect();

    log_info!("Processing {} URL(s)", urls.len());
    println!();

    // Shared counters and concurrency limiter
    let counters = Arc::new(Counters::new());
    let semaphore = Arc::new(Semaphore::new(args.parallel_jobs));
    let output_folder = Arc::new(args.output_folder);

    // Global rate limiter – one token bucket shared by all parallel workers so
    // the aggregate request rate never exceeds --rate-limit per hour.
    let quota = Quota::per_hour(
        NonZeroU32::new(args.rate_limit)
            .expect("rate_limit must be > 0"),
    );
    let limiter = Arc::new(RateLimiter::direct(quota));

    // Spawn one task per URL
    let mut handles = Vec::with_capacity(urls.len());
    for (index, url) in urls.into_iter().enumerate() {
        let client = Arc::clone(&client);
        let semaphore = Arc::clone(&semaphore);
        let counters = Arc::clone(&counters);
        let output_folder = Arc::clone(&output_folder);
        let limiter = Arc::clone(&limiter);

        let handle = tokio::spawn(async move {
            let _permit = semaphore.acquire().await.expect("semaphore closed");
            let result = download_image(&client, &url, &output_folder, index, &limiter).await;
            match result {
                DownloadResult::Success => {
                    counters.success.fetch_add(1, Ordering::Relaxed);
                }
                DownloadResult::Skipped => {
                    counters.skipped.fetch_add(1, Ordering::Relaxed);
                }
                DownloadResult::Failed => {
                    counters.failed.fetch_add(1, Ordering::Relaxed);
                }
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        let _ = handle.await;
    }

    // Summary
    let success = counters.success.load(Ordering::Relaxed);
    let failed = counters.failed.load(Ordering::Relaxed);
    let skipped = counters.skipped.load(Ordering::Relaxed);

    println!();
    println!("=============================================");
    log_info!("Download Summary:");
    println!("  {GREEN}✓ Successfully downloaded:{NC} {success}");
    println!("  {YELLOW}⊘ Skipped (already exist):{NC} {skipped}");
    println!("  {RED}✗ Failed:{NC} {failed}");
    println!("=============================================");

    Ok(())
}

