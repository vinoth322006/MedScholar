FROM python:3.11-slim

WORKDIR /app

# Build tools some ML packages need to compile native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install the CPU-only PyTorch build explicitly FIRST.
# Without this, pip pulls the default CUDA-enabled torch wheel, which is
# several GB and useless on a CPU-only deployment (HF Spaces free tier,
# most low-cost hosts) — this keeps the image small and the build fast.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Hugging Face Spaces (Docker SDK) requires the container to listen on 7860.
# If deploying elsewhere (Railway, Fly.io, your own server), change this
# port and the corresponding platform config to match — the app itself
# doesn't care what port it's told to bind to.
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]