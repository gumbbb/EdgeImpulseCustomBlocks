# Custom DSP Block: CAN Behavior Analysis

This block implements the second stage (Feature Engineering and Scaling) of the CAN Anomaly Detection pipeline. It takes 1Hz windowed data and extracts high-level behavioral features for the LGBM model.

## Overview

- **Protocol**: HTTP Server (Port 4446).
- **Architecture**: Based on the [official Edge Impulse Python DSP template](https://github.com/edgeimpulse/example-custom-processing-block-python).
- **Functionality**:
    1.  **Step 4 (Feature Engineering)**: Calculates TTC (Time to Collision), rolling averages, and behavioral flags (lane change, unsteady driving).
    2.  **Step 4 (Aggregation)**: Collapses the 30-second window into a **single 1x14 feature vector** using Mean (for features) and Max (for labels).
    3.  **Step 5 (Scaling)**: Applies `StandardScaler` normalization to the final vector.
    4.  **Output**: Returns a flattened array of 14 normalized features to the Studio.

## File Structure

- `dsp-server.py`: The official Edge Impulse HTTP boilerplate.
- `dsp.py`: The core `generate_features` implementation.
- `parameters.json`: Configuration for the Edge Impulse Studio UI.
- `Dockerfile`: Container definition (Exposes port 4446).
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

Follow the prompts to select your organization. Once pushed, it will be available as a new **Processing Block** in the **Create impulse** screen.

## Local Development

To run the DSP server locally for testing:

```bash
python dsp-server.py
```

The server will listen on `0.0.0.0:4446`. You can then use the Edge Impulse [Blocks Runner](https://docs.edgeimpulse.com/docs/edge-impulse-cli/edge-impulse-blocks#running-blocks-locally) to test it with real data from your project.
