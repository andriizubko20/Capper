# SStats API

**Category:** APIs & Data
**Created:** 2026-04-19
**Updated:** 2026-04-19

Основне джерело даних для Capper. Base URL: `https://api.sstats.net`. Auth через `X-API-KEY` header.

## Key Facts

- Client class: `SStatsClient` в `data/api_client.py`
- API key: env var `sstats_api_key`
- Retry logic з обробкою rate limit (429)
- Покриває 6 ліг: EPL, La Liga, Bundesliga, Serie A, Ligue 1, Champions League

## Endpoints

| Endpoint | Дані |
|---|---|
| `/Leagues` | Список ліг |
| `/Teams/list` | Команди ліги |
| `/Games/list` | Матчі/fixtures |
| `/Games/glicko/{fixture_id}` | Glicko-2 рейтинги + xG |
| `/Games/injuries` | Травми гравців |
| `/Odds/{fixture_id}` | Коефіцієнти |
| `/Odds/live-changes/{fixture_id}` | Рух лінії (odds movement) |

## Related

- [Capper Overview](capper-overview.md)
- [xG — Expected Goals](xg-expected-goals.md)

## Sources

- `data/api_client.py`, `config/settings.py`
