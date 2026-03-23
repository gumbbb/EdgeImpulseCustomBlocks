#!/bin/bash
 
BUCKET_PATH="${1:-/mnt/s3fs/fpt-edge-impulse}"
OUTPUT_DIR="${2:-./output}"
API_KEY="${3:-}"
 
if [ ! -d "$BUCKET_PATH" ]; then
    echo "Error: Input directory not found: $BUCKET_PATH"
    echo "Usage: $0 <bucket_path> [output_directory] [api_key]"
    exit 1
fi
 
mkdir -p "$OUTPUT_DIR"
 
echo "=========================================="
echo "CAN Anomaly Detection - Transformation Block"
echo "=========================================="
echo "Input path:       $BUCKET_PATH"
echo "Output directory:  $OUTPUT_DIR"
echo "Window size:       30"
echo "Skip times:        10"
echo "=========================================="
 
python3 transform.py \
    --bucket_name "$(basename "$BUCKET_PATH")" \
    --window_size 30 \
    --num_negative_samples 10 \
    --seq_len 30 \
    --skip_times 10 \
    --out-directory "$OUTPUT_DIR" \
    ${API_KEY:+--api_key "$API_KEY"}
 
echo ""
echo "=========================================="
echo "Transformation completed!"
echo "Output files saved to: $OUTPUT_DIR"
echo "=========================================="
 
 