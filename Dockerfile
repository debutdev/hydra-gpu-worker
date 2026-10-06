FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN python -m playwright install --with-deps chromium
COPY worker.py /app/worker.py
ENV HF_HOME=/runpod-volume/hydra/hf-cache PYTHONUNBUFFERED=1
CMD ["python", "-u", "/app/worker.py"]
