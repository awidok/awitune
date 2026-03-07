FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code
RUN pip install --no-cache-dir --break-system-packages \
    polars \
    scikit-learn \
    catboost \
    numpy \
    pandas \
    tensorboard \
    lightgbm \
    xgboost \
    optuna \
    flask \
    flask-cors \
    requests \
    python-dotenv \
    pyarrow
WORKDIR /app
RUN useradd -m -s /bin/bash agent && \
    mkdir -p /app/workspace /app/output /app/data && \
    chown -R agent:agent /app
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
USER agent
ENTRYPOINT ["/entrypoint.sh"]
