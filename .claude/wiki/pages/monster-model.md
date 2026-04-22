# Monster Model

**Category:** ML & Modeling
**Created:** 2026-04-19
**Updated:** 2026-04-19

Ансамбль моделей з різними порогами під кожну лігу окремо. Широке покриття, всі моделі перевірені на дистанції.

## Key Facts

- Ансамбль багатьох моделей
- Окремі пороги під кожну лігу (не одна модель для всіх)
- Широке покриття — більше ставок ніж у [WS Gap](ws-gap-model.md) і [Aquamarine](aquamarine-model.md)
- Всі включені моделі показали себе на дистанції
- Одна з трьох продакшн-моделей Capper

## Details

Monster — найширша з трьох моделей. Включає весь набір ліго-специфічних моделей що пройшли валідацію. Кожна ліга має власні пороги, що дозволяє адаптуватись до різної природи різних чемпіонатів.

[Aquamarine](aquamarine-model.md) — це підмножина Monster з найвищим winrate.

## Related

- [Aquamarine Model](aquamarine-model.md)
- [WS Gap Model](ws-gap-model.md)
- [Capper Overview](capper-overview.md)

## Sources

- `CLAUDE.md` — models section
