# Kelly Criterion

**Category:** Betting & EV
**Created:** 2026-04-19
**Updated:** 2026-04-19

Формула оптимального розміру ставки відносно банкролу. Єдиний механізм управління стейками в Capper.

## Key Facts

- Формула: `f = (p × b - q) / b`, де `p` — ймовірність перемоги, `q = 1 - p`, `b` — чистий виграш на одиницю
- Capper використовує **Fractional Kelly 25%** — множимо результат на 0.25
- Stop-loss не реалізований — Kelly управляє всіма ставками
- Вимагає відкаліброваних ймовірностей

## Details

Full Kelly максимізує довгостроковий ріст банкролу, але дає великі просідання при помилках моделі. Fractional Kelly (25%) зменшує ризик у 4 рази з незначною втратою в очікуваному рості.

Приклад: p=0.55, коефіцієнт 2.10 (b=1.10)
`f = (0.55 × 1.10 - 0.45) / 1.10 = (0.605 - 0.45) / 1.10 = 0.141`
Full Kelly → 14.1% банкролу. Fractional (25%) → 3.5% банкролу.

## Related

- [Expected Value (EV)](expected-value.md)
- [Capper Overview](capper-overview.md)

## Sources

- `CLAUDE.md` — key concepts section
