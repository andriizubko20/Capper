# Capper — команди для локальної розробки

# Запустити все (БД + scheduler + бот)
up:
	docker compose up -d

# Зупинити все
down:
	docker compose down

# Перезапустити все
restart:
	docker compose down && docker compose up -d

# Переглянути логи (всі сервіси)
logs:
	docker compose logs -f

# Логи конкретного сервісу: make logs-scheduler
logs-scheduler:
	docker compose logs -f scheduler

logs-bot:
	docker compose logs -f bot

logs-db:
	docker compose logs -f db

# Застосувати міграції БД
migrate:
	docker compose exec scheduler alembic upgrade head

# Створити нову міграцію: make migration name="add_something"
migration:
	docker compose exec scheduler alembic revision --autogenerate -m "$(name)"

# Ретрейнінг моделі вручну
retrain:
	docker compose exec scheduler python -m scheduler.tasks.retrain

# Зібрати дані вручну
collect:
	docker compose exec scheduler python -m scheduler.tasks.collect_data

# Згенерувати пики вручну
picks:
	docker compose exec scheduler python -m scheduler.tasks.generate_picks

# Перебудувати Docker образи після змін в коді
build:
	docker compose build

# Статус контейнерів
status:
	docker compose ps
