# claude-hackathon-agent

Alma's FastAPI backend — LangChain LCEL pipeline with Redis semantic cache, injection guard, and MCP memory client.

> Part of the **Alma** system — an emotional AI companion for mental health. See [claude-hackathon-alma](https://github.com/iDeepBrain/claude-hackathon-alma) for the full architecture, diagrams, and project overview.

## Prerequisites

- Docker + Docker Compose
- `claude-hackathon-mcp` repo cloned as a sibling directory (docker-compose builds it)

```
Claude-Hackathon-Opus/
├── claude-hackathon-agent/   ← you are here
└── claude-hackathon-mcp/
```

## Setup

Copy the env file and add your Anthropic API key:

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

`.env.example`:
```
ANTHROPIC_API_KEY=sk-ant-...
REDIS_URL=redis://redis:6379
MCP_URL=http://mcp:8001/mcp
SESSION_TTL=86400
CACHE_TTL=3600
CACHE_THRESHOLD=0.92
```

Do the same for the MCP repo:
```bash
cp ../claude-hackathon-mcp/.env.example ../claude-hackathon-mcp/.env
```

---

## Docker commands

All commands run from this directory (`claude-hackathon-agent/`).

### Build

Build all images (agent + mcp). Run this after changing `requirements.txt` or `Dockerfile`.

```bash
docker compose build
```

Build a single service:

```bash
docker compose build agent
docker compose build mcp
```

Force a clean rebuild (no cache):

```bash
docker compose build --no-cache
```

---

### Start

Start all services (agent, mcp, redis) in the background:

```bash
docker compose up -d
```

Start with logs visible:

```bash
docker compose up
```

Start a single service:

```bash
docker compose up -d agent
```

Stop everything:

```bash
docker compose down
```

Stop and remove volumes (clears Redis + SQLite data):

```bash
docker compose down -v
```

---

### Test

Run the agent test suite:

```bash
docker compose exec agent pytest tests/ -v
```

Run only a specific test file:

```bash
docker compose exec agent pytest tests/test_guard.py -v
docker compose exec agent pytest tests/test_chain.py -v
docker compose exec agent pytest tests/test_csv.py -v
```

Run the MCP test suite:

```bash
docker compose exec mcp pytest tests/ -v
```

Run both suites (quick summary):

```bash
docker compose exec agent pytest tests/ -q && docker compose exec mcp pytest tests/ -q
```

---

### Useful extras

View logs:

```bash
docker compose logs -f agent
docker compose logs -f mcp
docker compose logs -f redis
```

Open a shell inside a container:

```bash
docker compose exec agent bash
docker compose exec mcp bash
```

Check Redis keys:

```bash
docker compose exec redis redis-cli keys "*"
```

---

## API endpoints

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `POST` | `/api/v1/chat` | `{user_id, message, language?, image_base64?}` | Stream SSE response from Alma |
| `POST` | `/api/v1/trigger` | `{user_id, trigger_type}` | Proactive check-in (morning_checkin, silence_anomaly, event_followup) |
| `GET` | `/health` | — | Health check |

**Example chat request:**

```bash
curl -N -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u123", "message": "hola Alma", "language": "es"}'
```

**Example trigger:**

```bash
curl -N -X POST http://localhost:8080/api/v1/trigger \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u123", "trigger_type": "morning_checkin"}'
```

---

Built by [Cristian Lazo Quispe](https://github.com/CristianLazoQuispe). Licensed under MIT (© iDeepBrain).
