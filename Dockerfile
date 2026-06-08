# Mafia Simulator -- WebSocket game server, ready for Cloud Run.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so they're cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects $PORT (defaults to 8080 locally) and expects the
# container to listen on 0.0.0.0; `exec` keeps uvicorn as PID 1 so it
# receives SIGTERM directly during deploys/scale-down.
ENV PORT=8080
EXPOSE 8080
CMD exec uvicorn server:app --host 0.0.0.0 --port ${PORT}
