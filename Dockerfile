# MindCare production image.
#
# Built for free hosting on Hugging Face Spaces (Docker SDK). Runs the
# classifier + Qwen 2.5 generator on CPU (Spaces' free CPU Basic tier gives
# 2 vCPU / 16GB RAM, which comfortably fits both models -- no GPU or paid
# tier needed). See DEPLOYMENT.md for the full walkthrough.
#
# Two things make this specifically Spaces-friendly:
#   1. Listens on $PORT, defaulting to 7860 (the port Spaces requires).
#   2. Model weights are downloaded and baked into the image at BUILD time
#      (see the RUN step below), not fetched at runtime. Free Spaces have
#      ephemeral storage -- anything written while the container is running
#      is lost when the Space sleeps and wakes back up. Baking weights into
#      an image layer means they survive that cycle with no re-download.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached across code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model cache lives inside the image itself (baked in below), not on the
# ephemeral runtime disk -- this is what makes it survive Space sleep/wake.
ENV HF_HOME=/app/hf-cache
ENV TRANSFORMERS_CACHE=/app/hf-cache
ENV MINDCARE_GENERATOR_MODEL=Qwen/Qwen2.5-1.5B-Instruct

# Pre-download both the generator and the base classifier backbone at BUILD
# time. This makes the build slower (several minutes) but means the running
# container never needs network access to load a model, and cold starts
# after a Space wakes from sleep are fast instead of "redownload 3GB".
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')"
RUN python -c "\
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification; \
DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased'); \
DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')"

# Now bring in the app code -- and, importantly, your trained classifier
# checkpoint at models/distilbert-intent/ if you've run train_distilbert.py
# and committed it (see DEPLOYMENT.md). This layer changes most often so it
# goes last, after the slow model-download layers above.
COPY . .

RUN mkdir -p /data

# SQLite lives here. On free Spaces this directory is ephemeral (wiped on
# sleep/rebuild) since there's no persistent volume on the free tier --
# see DEPLOYMENT.md for what that trade-off means and how to avoid it if
# it matters for your testing period.
ENV MINDCARE_DB_PATH=/data/mindcare.db

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD sh -c 'curl -f http://localhost:${PORT:-7860}/api/health || exit 1'

# Single worker: the models are loaded once per process, and duplicating
# them across multiple workers multiplies memory use for no benefit at
# small-group scale. --threads lets Flask still serve several requests
# concurrently; model inference itself queues, which is fine for a handful
# of simultaneous testers. Binds to $PORT if the platform sets one (Render
# does), otherwise defaults to 7860 (what Spaces requires).
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 120 app:app"]
