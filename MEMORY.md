# Engineering Notes

Рабочие заметки Claude по проекту Capper. Сюда идут технические наблюдения: ошибки и их решения, особенности команд, архитектурные детали которые не очевидны из кода.

---

## Architecture Notes

_Пока пусто — заметки появятся в процессе разработки._

---

## Errors & Solutions

- **pydantic extra fields error** — `.env` містить змінні для Docker (`POSTGRES_USER` і т.д.) яких нема в `Settings`. Фікс: `extra = "ignore"` в `Config` класі `settings.py`.
- **`python` not found** — на цьому Mac використовується `python3`. У venv після активації працює як `python`.

---

## Build & Run Notes

```bash
# Активувати venv
source venv/bin/activate

# Підняти БД
docker compose up -d

# Створити міграцію після змін в models.py
alembic revision --autogenerate -m "опис змін"

# Застосувати міграції
alembic upgrade head
```

pgAdmin доступний на http://localhost:5050 (admin@capper.com / admin).
