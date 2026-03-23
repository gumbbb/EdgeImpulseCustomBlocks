FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy block files and dependencies
COPY . .

# Edge Impulse expects DSP blocks to listen on port 4446 per the standard example
EXPOSE 4446

ENTRYPOINT ["python3", "dsp-server.py"]
