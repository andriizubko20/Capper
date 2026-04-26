# Wiki Index

_Content catalog. Updated automatically on every INGEST or LINT operation._

## Betting & EV

| Page | Summary |
|---|---|
| [Expected Value (EV)](pages/expected-value.md) | EV = (p × odds) - 1. Основний фільтр ставок. |
| [Closing Line Value (CLV)](pages/closing-line-value.md) | Метрика довгострокової якості моделі. |
| [Kelly Criterion](pages/kelly-criterion.md) | Оптимальний розмір ставки. Fractional 25% у Capper. |

## ML & Modeling

| Page | Summary |
|---|---|
| [WS Gap Model](pages/ws-gap-model.md) | 2-рівневий фільтр. Мало ставок, висока якість. |
| [Monster Model](pages/monster-model.md) | Ансамбль з порогами під кожну лігу. Широке покриття. |
| [Aquamarine Model](pages/aquamarine-model.md) | Підмножина Monster. Тільки найвищий winrate. |
| [Gem Model (v1)](pages/gem-model.md) | 3-model stacking + calibration. WR≥70%, flat 4%. Phase 3 code done, smoke passed. |
| [Gem Roadmap 2 Weeks](pages/gem-model-roadmap.md) | Day-by-day план 25.04→08.05: train → forward-test → telegram → deploy. |
| [Gem Day 1 Analysis](pages/gem-analysis-2026-04-25.md) | Аналіз 1-го тренінгу: WR 70.3% але yield 1/wk; gem_score >20% → 85.7%; рекомендації для v2. |
| [Work Plan 2026-04-26](pages/work-plan-2026-04-26.md) | 6 blocks today: Self-Glicko, opening odds collector, injuries, lineups, deploy Gem v2, monitoring. ~8h. |
| [Session 2026-04-26 — Infra Refactor](pages/session-2026-04-26.md) | Per-league cherry-pick devig, Kelly compounding fix, 3 Quick Wins (bookmaker shopping/CLV monitor/movement filter), Mac proxy fix + ProxyMon. |

## Football

| Page | Summary |
|---|---|
| [xG — Expected Goals](pages/xg-expected-goals.md) | Якість атаки/захисту без шуму реальних голів. |
| [Elo Ratings](pages/elo-ratings.md) | Динамічна оцінка сили команд. |

## APIs & Data

| Page | Summary |
|---|---|
| [SStats API](pages/sstats-api.md) | `api.sstats.net`. Glicko-2, xG, odds, injuries. Top-5 + UCL. |

## Tools & Tech

| Page | Summary |
|---|---|
| [Capper Overview](pages/capper-overview.md) | Архітектура, стек, pipeline проекту. |
| [Project Status](pages/project-status.md) | Поточний стан: bug fixes 2026-04-22, pending medium issues, deploy. |
| [Bug Audit 2026-04-22](pages/bug-audit-2026-04-22.md) | 33 знайдених проблеми (critical+high виправлені); 12 medium залишились задокументованими. |
| [VPS SStats Proxy](pages/vps-sstats-proxy.md) | Workaround DigitalOcean ↔ sstats.net: Mac proxy + SSH reverse tunnel + socat. Temp solution — Mac dependency. |
| [DB Restore 2026-04-22](pages/db-restore-2026-04-22.md) | Schema drift: match_stats/injuries/monster_p_is відсутні на VPS; відновлено з локальної БД (pg_dump + stream). |

## Research

| Page | Summary |
|---|---|
| _(empty)_ | |

## People

| Page | Summary |
|---|---|
| _(empty)_ | |
