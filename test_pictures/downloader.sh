#!/bin/bash

# =============================================================================
# Image Downloader Script
# Downloads images from a TSV file containing URLs
# =============================================================================

set -euo pipefail

# Configuration
MAX_RETRIES=5
RETRY_DELAY=5
TIMEOUT=30
PARALLEL_JOBS=5

# Rate limiting configuration
RATE_LIMIT_DELAY=1.0  # Delay between requests in seconds (1 second)
REQUEST_BURST_SIZE=10 # Number of requests before enforcing a longer delay
BURST_DELAY=5         # Longer delay after burst (in seconds)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters (using temp files for parallel processing)
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Default TSV file URL and local path
TSV_URL="https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-test.tsv"
DEFAULT_TSV_FILE="$(dirname "$0")/open-images-dataset-test.tsv"
OUTPUT_FOLDER="$(dirname "$0")/openimage"

# Display usage information
usage() {
    echo -e "${BLUE}Usage:${NC} $0 <num_images> [parallel_jobs]"
    echo ""
    echo "Arguments:"
    echo "  <num_images>     Maximum number of images to download"
    echo "  [parallel_jobs]  Optional: Number of parallel downloads (default: $PARALLEL_JOBS)"
    echo ""
    echo "Output folder: $OUTPUT_FOLDER (hard-coded)"
    echo ""
    echo "Example: $0 100 10"
    echo ""
    echo "The script automatically downloads the Open Images Dataset TSV file if not present."
    echo "TSV Source: $TSV_URL"
    echo ""
    echo "Requirements: curl"
    exit 1
}

# Download TSV file if not present
download_tsv_if_needed() {
    if [ -f "$DEFAULT_TSV_FILE" ] && [ -s "$DEFAULT_TSV_FILE" ]; then
        log_info "TSV file already exists: $DEFAULT_TSV_FILE"
        return 0
    fi

    log_info "Downloading Open Images Dataset TSV file..."
    log_info "Source: $TSV_URL"

    if curl -L --progress-bar --fail -o "$DEFAULT_TSV_FILE" "$TSV_URL"; then
        log_success "TSV file downloaded successfully"
        return 0
    else
        log_error "Failed to download TSV file from: $TSV_URL"
        rm -f "$DEFAULT_TSV_FILE"
        exit 1
    fi
}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Download a single image with retry logic
download_image() {
    local url="$1"
    local output_folder="$2"
    local index="$3"

    # Extract filename from URL (remove query parameters)
    local filename
    filename=$(basename "$(echo "$url" | cut -d'?' -f1)")

    # Generate unique filename if extraction fails
    if [ -z "$filename" ] || [ "$filename" == "/" ]; then
        filename="image_${index}_$(date +%s%N).jpg"
    fi

    local file_path="$output_folder/$filename"

    # Skip if file already exists and is not empty (local check only, no server access)
    if [ -f "$file_path" ] && [ -s "$file_path" ]; then
        log_warning "File already exists, skipping: $filename"
        echo "skipped" >> "$TEMP_DIR/skipped.count"
        return 0
    fi

    # Rate limiting: Add delay before download
    sleep "$RATE_LIMIT_DELAY"

    # Add longer delay after burst
    if [ $((index % REQUEST_BURST_SIZE)) -eq 0 ] && [ "$index" -gt 0 ]; then
        sleep "$BURST_DELAY"
    fi

    # Download with retry logic
    local attempt=1
    while [ $attempt -le $MAX_RETRIES ]; do
        # Capture HTTP status code
        local http_code
        http_code=$(curl -s -L \
            --max-time "$TIMEOUT" \
            --retry 0 \
            --output "$file_path" \
            --write-out "%{http_code}" \
            "$url" 2>/dev/null || echo "000")

        # Check for specific HTTP status codes
        if [ "$http_code" == "404" ]; then
            log_warning "Image not found (404), skipping: $filename"
            rm -f "$file_path"
            echo "skipped" >> "$TEMP_DIR/skipped.count"
            return 0
        fi

        if [ "$http_code" == "410" ]; then
            log_warning "Image permanently deleted (410 Gone), skipping: $filename"
            rm -f "$file_path"
            echo "skipped" >> "$TEMP_DIR/skipped.count"
            return 0
        fi

        if [ "$http_code" == "429" ]; then
            # Rate limited - wait longer before retry
            local rate_limit_wait=$((RETRY_DELAY * 2 * attempt))
            log_warning "Rate limited (429), waiting ${rate_limit_wait}s before retry $attempt/$MAX_RETRIES: $filename"
            rm -f "$file_path"
            sleep "$rate_limit_wait"
            attempt=$((attempt + 1))
            continue
        fi

        # Check if download was successful (2xx status codes)
        if [[ "$http_code" =~ ^2[0-9][0-9]$ ]]; then
            # Verify the downloaded file is valid (not empty and is an image)
            if [ -s "$file_path" ]; then
                # Check if file is a valid image using file command
                local file_type
                file_type=$(file -b --mime-type "$file_path" 2>/dev/null || echo "unknown")

                if [[ "$file_type" == image/* ]]; then
                    log_success "Downloaded: $filename (HTTP $http_code)"
                    echo "success" >> "$TEMP_DIR/success.count"
                    return 0
                else
                    log_warning "Invalid image type ($file_type): $filename"
                    rm -f "$file_path"
                fi
            else
                rm -f "$file_path"
            fi
        else
            # Other error codes (5xx, 3xx, etc.)
            log_warning "HTTP $http_code for: $filename (attempt $attempt/$MAX_RETRIES)"
            rm -f "$file_path"
        fi

        if [ $attempt -lt $MAX_RETRIES ]; then
            sleep "$RETRY_DELAY"
        fi
        attempt=$((attempt + 1))
    done

    log_error "Failed after $MAX_RETRIES attempts (last HTTP code: $http_code): $url"
    echo "failed" >> "$TEMP_DIR/failed.count"
    return 1
}

# Export functions for parallel execution
export -f download_image log_info log_success log_warning log_error
export RED GREEN YELLOW BLUE NC MAX_RETRIES RETRY_DELAY TIMEOUT TEMP_DIR
export RATE_LIMIT_DELAY REQUEST_BURST_SIZE BURST_DELAY

# Validate arguments
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage
fi

NUM_IMAGES="$1"
PARALLEL_JOBS="${2:-$PARALLEL_JOBS}"

# Check dependencies
if ! command -v curl &> /dev/null; then
    log_error "'curl' is not installed. Please install it to run this script."
    exit 1
fi

# Download TSV file if needed
download_tsv_if_needed
TSV_FILE="$DEFAULT_TSV_FILE"

# Validate num_images is a positive number
if ! [[ "$NUM_IMAGES" =~ ^[0-9]+$ ]] || [ "$NUM_IMAGES" -le 0 ]; then
    log_error "Number of images must be a positive integer"
    exit 1
fi

# Create output folder
mkdir -p "$OUTPUT_FOLDER"

log_info "Starting download of up to $NUM_IMAGES images"
log_info "Source: $TSV_FILE"
log_info "Destination: $OUTPUT_FOLDER"
log_info "Parallel jobs: $PARALLEL_JOBS"
echo ""

# Initialize counter files
touch "$TEMP_DIR/success.count" "$TEMP_DIR/failed.count" "$TEMP_DIR/skipped.count"

# Extract valid URLs and process them
# - Skip header line (tail -n +2)
# - Extract first column (URL)
# - Filter valid HTTP(S) URLs
# - Limit to requested number
INDEX=0
tail -n +2 "$TSV_FILE" | while IFS=$'\t' read -r url rest_of_line; do
    if [ "$INDEX" -ge "$NUM_IMAGES" ]; then
        break
    fi

    if [[ "$url" =~ ^https?:// ]]; then
        echo "$url $OUTPUT_FOLDER $INDEX"
        INDEX=$((INDEX + 1))
    else
        log_warning "Invalid URL format, skipping: ${url:0:50}..."
    fi
done | xargs -P "$PARALLEL_JOBS" -L 1 bash -c 'download_image "$1" "$2" "$3"' _

# Calculate final statistics
SUCCESS_COUNT=$(wc -l < "$TEMP_DIR/success.count" 2>/dev/null | tr -d ' ' || echo 0)
FAILED_COUNT=$(wc -l < "$TEMP_DIR/failed.count" 2>/dev/null | tr -d ' ' || echo 0)
SKIPPED_COUNT=$(wc -l < "$TEMP_DIR/skipped.count" 2>/dev/null | tr -d ' ' || echo 0)

echo ""
echo "============================================="
log_info "Download Summary:"
echo -e "  ${GREEN}✓ Successfully downloaded:${NC} $SUCCESS_COUNT"
echo -e "  ${YELLOW}⊘ Skipped (already exist):${NC} $SKIPPED_COUNT"
echo -e "  ${RED}✗ Failed:${NC} $FAILED_COUNT"
echo "============================================="
