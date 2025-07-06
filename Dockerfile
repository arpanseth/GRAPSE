FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install magic-pdf dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY environment.yml .
RUN pip install --no-cache-dir \
    aiohappyeyeballs==2.4.4 \
    aiohttp==3.11.11 \
    aiosignal==1.3.2 \
    annotated-types==0.7.0 \
    anyio==4.8.0 \
    attrs==25.1.0 \
    boto3==1.36.8 \
    botocore==1.36.8 \
    brotli==1.1.0 \
    certifi==2024.12.14 \
    cffi==1.17.1 \
    charset-normalizer==3.4.1 \
    click==8.1.8 \
    colorlog==6.9.0 \
    cryptography==44.0.0 \
    dataclasses-json==0.6.7 \
    distro==1.9.0 \
    fast-langdetect==0.2.5 \
    fasttext-predict==0.9.2.4 \
    frozenlist==1.5.0 \
    h11==0.14.0 \
    httpcore==1.0.7 \
    httpx==0.28.1 \
    httpx-sse==0.4.0 \
    idna==3.10 \
    jiter==0.8.2 \
    jmespath==1.0.1 \
    joblib==1.4.2 \
    jsonpatch==1.33 \
    jsonpointer==3.0.0 \
    langchain==0.3.16 \
    langchain-community==0.3.16 \
    langchain-core==0.3.32 \
    langchain-text-splitters==0.3.5 \
    langsmith==0.3.2 \
    loguru==0.7.3 \
    magic-pdf==0.6.1 \
    marshmallow==3.26.0 \
    multidict==6.1.0 \
    mypy-extensions==1.0.0 \
    numpy==1.26.4 \
    openai==1.60.2 \
    orjson==3.10.15 \
    packaging==24.2 \
    pdfminer-six==20240706 \
    propcache==0.2.1 \
    pycparser==2.22 \
    pydantic==2.10.6 \
    pydantic-core==2.27.2 \
    pydantic-settings==2.7.1 \
    pymupdf==1.25.2 \
    python-dateutil==2.9.0.post0 \
    python-dotenv==1.0.1 \
    pyyaml==6.0.2 \
    requests==2.32.3 \
    requests-toolbelt==1.0.0 \
    robust-downloader==0.0.2 \
    s3transfer==0.11.2 \
    scikit-learn==1.6.1 \
    scipy==1.15.1 \
    six==1.17.0 \
    sniffio==1.3.1 \
    sqlalchemy==2.0.37 \
    tenacity==9.0.0 \
    threadpoolctl==3.5.0 \
    tqdm==4.67.1 \
    typing-extensions==4.12.2 \
    typing-inspect==0.9.0 \
    urllib3==2.3.0 \
    wordninja==2.0.0 \
    yarl==1.18.3 \
    zstandard==0.23.0 \
    watchdog==3.0.0 \
    nltk==3.8.1

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p papers output state logs

# Create processed papers state file
RUN echo '{}' > /app/state/processed_papers.json

# Make scripts executable
RUN chmod +x scripts/*.py
RUN find . -name "*.sh" -exec chmod +x {} \; || true

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt_tab')"

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for web interface (if implemented)
EXPOSE 8000

# Health check - simple Python import test
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "scripts/docker-entrypoint.py"]