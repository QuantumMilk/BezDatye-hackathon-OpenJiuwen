FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /requirements.lock
RUN pip install --no-cache-dir --prefix=/install -r /requirements.lock

FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
ENV PYTHONDONTWRITEBYTECODE=1 OPENJIUWEN_RUNTIME_LOG_DIR=/tmp
COPY --from=dependencies /install /usr/local
WORKDIR /app
COPY app/harness/ harness/
COPY app/run_agent.py run_agent.py
COPY app/scripts/check_api.py scripts/check_api.py
COPY app/scripts/check_runtime.py scripts/check_runtime.py
ENTRYPOINT ["python", "/app/run_agent.py"]
