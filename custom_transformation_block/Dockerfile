FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy block files and dependencies (src, transform.py, parameters.json)
COPY . .

# Edge Impulse passes CLI arguments directly to the entrypoint
ENTRYPOINT ["python3", "transform.py"]
