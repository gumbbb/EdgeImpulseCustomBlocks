FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip3 --no-cache-dir install -r requirements.txt

COPY . ./

VOLUME /data

ENTRYPOINT ["python3", "transform.py"]