# claude-hackathon-agent

FastAPI + AlmaChain — recibe mensajes, llama a Claude vía Anthropic SDK, usa MCP para memoria, y envía mensajes proactivos a Telegram via APScheduler.

## Para el stack completo (recomendado)

```bash
cd ../claude-hackathon-infra
docker compose up --build -d
```

**No levantar ambos composes simultáneamente** — conflicto en puerto 8080 y Redis duplicado.

## Para desarrollo standalone (solo si infra NO está corriendo)

```bash
cd claude-hackathon-agent
docker compose -f docker-compose.standalone.yml up --build -d
```

Levanta: `redis` (6379), `mcp` (8082), `agent` (8080).

> El compose se renombró a `.standalone.yml` para evitar que `docker compose up` lo
> tome por defecto y conflicte con el stack de `claude-hackathon-infra` (mismos puertos).

## API endpoints

```
GET  /health                       → {"status":"ok","model":"haiku"}
POST /api/v1/chat                  → SSE stream de respuesta (AlmaChain)
GET  /api/v1/memory/{user_id}      → 4 capas de memoria del usuario (JSON)
POST /api/v1/trigger               → Disparar mensaje proactivo manual (dev)
POST /api/v1/demo/seed             → Re-seed demo user "Mateo" con memoria 7-day (X-Demo-Token requerido)
POST /api/v1/auth/google           → Verifica id_token de Google Identity Services, retorna user_id "google_<sub>"
GET  /api/v1/config                → Public client config (Google OAuth Client ID si está configurado)
POST /cron/proactive/{slot}        → Cloud Scheduler endpoint (X-Cloud-Scheduler-Token)
```

## AlmaChain pipeline

```
1. is_injection(message)           → bloquear prompt injection
2. semantic_cache.lookup(message)  → retornar cached si hit (rapidfuzz + cosine)
3. mcp_client.build_context(uid)   → inyectar memoria en system prompt
4. router.select_model(state)      → Haiku / Sonnet / Opus según contexto
5. llm.astream(messages)           → streaming SSE
6. asyncio.create_task(post_response_work):
   ├── trim session history (max 40 msgs)
   ├── evaluate_crisis_risk
   ├── state transitions (onboarding → chat → crisis)
   ├── upsert mood_history
   ├── detect & upsert mentioned_events
   └── cache the response
```

## Model routing

| Condición | Modelo |
|-----------|--------|
| `crisis_score > 0.7` | `gemini-1.5-pro` |
| `has_image` | `gemini-2.0-flash` (vision) |
| `len(message) > 800` | `gemini-1.5-pro` |
| default | `gemini-2.0-flash` |

## Proactividad (APScheduler)

3 check-ins diarios enviados via httpx directo a Telegram Bot API:

| Slot | Hora Lima (UTC-5) | Mensaje |
|------|-------------------|---------|
| Desayuno | 08:30 | "¿Ya desayunaste? ☀️" |
| Almuerzo | 13:30 | "¿Ya almorzaste? 🌞 ¿Cómo va tu día?" |
| Cena | 19:30 | "¿Ya cenaste? 🌙 ¿Hiciste algo de movimiento hoy?" |

Gates: `crisis_score > 0.6` → suprimir | usuario activo últimas 2h → suprimir | slot ya enviado hoy → suprimir

Redis keys para proactividad:
```
alma:chat:{tg_user_id}                      → chat_id de Telegram
alma:proactive:last:{user_id}               → timestamp último mensaje proactivo
alma:proactive:slot:{user_id}:{date}:{slot} → flag "slot ya enviado"
```

## .env requerido

```
ANTHROPIC_API_KEY=sk-ant-...
REDIS_URL=redis://redis:6379
MCP_URL=http://mcp:8001/mcp
SESSION_TTL=86400
CACHE_TTL=3600
CACHE_THRESHOLD=0.92

# Proactividad
TELEGRAM_BOT_TOKEN=<token de BotFather>
SCHEDULER_ENABLED=true
PROACTIVE_TZ=America/Lima
PROACTIVE_BREAKFAST_H=8
PROACTIVE_LUNCH_H=13
PROACTIVE_DINNER_H=19
PROACTIVE_SILENCE_WINDOW_H=2

# Demo seed (opcional — solo activar si querés exponer /api/v1/demo/seed)
DEMO_SEED_TOKEN=<token compartido con quien dispare el reset>

# Google OAuth (opcional — sin esto, el CTA "Continuar con Google" no aparece en el frontend)
# Crear en Google Cloud Console → APIs & Services → Credentials → OAuth 2.0 Client (type Web)
# Authorized JavaScript origins: https://alma-bot.com, http://localhost:3000
GOOGLE_OAUTH_CLIENT_ID=<your-id>.apps.googleusercontent.com

# Cloud Scheduler auth (producción)
CRON_TOKEN=<token compartido con Cloud Scheduler>
```

## Test rápido

```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "message": "Hola Alma", "language": "es"}' -N
```

## Tests

```bash
pytest tests/
```

## Demo seed (resetear memoria de Mateo)

Para que el panel de memoria no se vea vacío en demos en frío:

```bash
# CLI local (requiere MCP corriendo en localhost:8001)
python scripts/reset_demo.py

# Producción vía endpoint protegido
curl -X POST https://alma-bot.com/api/v1/demo/seed \
  -H "X-Demo-Token: $DEMO_SEED_TOKEN"
```

Pobla las 4 capas de memoria de `demo_mateo` con 7 días de mood_history, eventos
mencionados (cita médica viernes, reunión con jefe), hábitos (sueño irregular,
sedentarismo), y preferencias de interacción. Idempotente.

Lógica en `app/seed_demo.py` — la persona Mateo está documentada ahí.

## Pendiente por implementar

| Item | Story | Sprint | Estado |
|------|-------|--------|--------|
| (vacío — todo implementado) | — | — | — |

## Qué ya funciona

- AlmaChain pipeline completo (6 pasos)
- SSE streaming POST /api/v1/chat
- Memory endpoint GET /api/v1/memory/{user_id}
- Trigger endpoint POST /api/v1/trigger (manual)
- MCP client con 5 métodos
- Semantic cache (rapidfuzz + cosine)
- Injection guard
- Model routing (Haiku/Sonnet/Opus — crisis usa Opus)
- APScheduler proactividad (3 cron jobs: breakfast/lunch/dinner)
- Redis keys `last_activity` y `crisis:last` escritas en `_post_response`
- Env vars proactividad en .env
- Validación de BOT_TOKEN al crear scheduler
