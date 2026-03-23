# Custom Transformation Block: CAN Anomaly Detection

This block implements the first stage of the CAN Anomaly Detection pipeline, following the "Fused Architecture" design. It handles raw data ingestion, resampling, windowing, and leakage prevention.

## Overview

- **Mode**: Standalone (scans buckets directly).
- **Functionality**: 
    1.  **Step 1 (Discovery)**: Locates `.csv.gz` files in mounted buckets (`/mnt/azure`, `/mnt/s3fs`) or a specified `--bucket_name`.
    2.  **Step 2 (Processing)**: 
        - Resamples raw 20Hz data to **1Hz**.
        - Creates `trip_id` and filters for anomaly events (PCS).
        - Generates 30-second windows.
        - **Label Leakage Prevention ("The Cut")**: Removes the last `skip_times + 1` rows from every window to prevent training on the actual anomaly signal.
    3.  **Output**: Saves cleaned 1Hz CSV files and an `ei-metadata.json` for Edge Impulse.

## File Structure

- `transform.py`: The main processing script.
- `parameters.json`: Configuration for the Edge Impulse Studio UI.
- `Dockerfile`: Container definition for Edge Impulse.
- `src/`: Locally encapsulated dependencies from the main project.
- `requirements.txt`: Python dependencies.

## Deployment

### Prerequisites

1.  [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/installation) installed.
2.  [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running.

### Push to Edge Impulse

From this directory, run:

```bash
edge-impulse-blocks push
```

Follow the prompts to select your organization and name the block. Once pushed, it will be available under **Data acquisition > Custom transformation** in your Edge Impulse project.

## Local Testing

You can simulate the block execution locally:

```bash
# Set input/output paths (ensure they exist)
python transform.py --bucket_name ./test_data --out-directory ./out
```
