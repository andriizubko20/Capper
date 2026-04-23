# DB Restore — 2026-04-22

**Category:** Tools & Tech
**Created:** 2026-04-22

Восстановлення історичних даних VPS з локальної БД після schema drift.

## Що сталося

VPS БД (`capper`, Postgres 16) знаходилась у частково пошкодженому стані:

| Таблиця | VPS (до) | Локально | Проблема |
|---|---|---|---|
| `match_stats` | 0 | 18,506 | повністю порожня |
| `injury_reports` | NOT EXISTS | 30,085 | таблиці не існувало |
| `monster_p_is` | NOT EXISTS | 76 | таблиці не існувало |
| `odds.opening_value` | NOT EXISTS | OK | колонка відсутня, блокувала весь SELECT |
| `match_stats.home/away_fouls` | NOT EXISTS | OK | колонки відсутні |
| `match_stats.home/away_offsides` | NOT EXISTS | OK | колонки відсутні |

**Наслідок:** моделі не могли генерувати нові піки — `_load_stats_df` повертав порожній DataFrame → `KeyError: 'home_team_id'` у `compute_xg_features`. Всі 57 активних piks були старі (до дрейфу).

## Що зроблено

### 0. Leagues constraint + 56 missing leagues (додано після першого раунду)

VPS мав стару `UNIQUE(api_id)` (схема до multi-season), локально — `UNIQUE(api_id, season)`. Фікс:
```sql
ALTER TABLE leagues DROP CONSTRAINT IF EXISTS leagues_api_id_key;
ALTER TABLE leagues ADD CONSTRAINT leagues_api_id_season_key UNIQUE (api_id, season);
```

Після цього — 11,485 orphaned matches (`matches.league_id` посилались на leagues що не існували). Додано 40 ліг для orphaned FKs + 16 сезон-2026. **Leagues: 30 → 86, orphaned matches: 11485 → 0.**

### 1. Створено відсутні таблиці/колонки

```sql
-- колонки
ALTER TABLE odds ADD COLUMN IF NOT EXISTS opening_value DOUBLE PRECISION;
ALTER TABLE match_stats ADD COLUMN IF NOT EXISTS home_fouls INTEGER;
ALTER TABLE match_stats ADD COLUMN IF NOT EXISTS away_fouls INTEGER;
ALTER TABLE match_stats ADD COLUMN IF NOT EXISTS home_offsides INTEGER;
ALTER TABLE match_stats ADD COLUMN IF NOT EXISTS away_offsides INTEGER;

-- таблиці
CREATE TABLE IF NOT EXISTS injury_reports (
  id SERIAL PRIMARY KEY,
  match_id INTEGER NOT NULL REFERENCES matches(id),
  team_id INTEGER NOT NULL REFERENCES teams(id),
  player_api_id INTEGER NOT NULL,
  player_name VARCHAR(150) NOT NULL,
  reason VARCHAR(255),
  UNIQUE(match_id, team_id, player_api_id)
);

-- monster_p_is створилась через pg_dump, бо там був CREATE TABLE
```

### 2. Перенесено історичні дані з локальної БД

**Перевірка identity match:** `matches.id` на локальній і VPS БД збігаються 1-в-1 (ID 462 → api_id 1299010 ідентично). Тому FK-посилання `match_id` переносяться без мапінгу.

**Процес:**
```bash
# дамп локально
docker exec capper_db pg_dump -U capper -d capper \
  -t match_stats -t injury_reports --data-only --column-inserts \
  > /tmp/stats_dump.sql  # 18 MB, 48k INSERTs

docker exec capper_db pg_dump -U capper -d capper \
  -t monster_p_is --column-inserts > /tmp/monster_p_is.sql

# трансфер і завантаження на VPS
scp /tmp/stats_dump.sql root@165.227.164.220:/tmp/
scp /tmp/monster_p_is.sql root@165.227.164.220:/tmp/

ssh root@165.227.164.220 "cat /tmp/stats_dump.sql | \
  docker compose -f /opt/capper/docker-compose.yml exec -T db \
  psql -U capper -d capper -q"
```

Стрім `psql | docker exec -T` обробляє по одному INSERT, жодних memory-piks на 1 GB VPS.

## Результат

```
leagues:        30 → 86  ✅ (+56, схема constraint пофіксена)
match_stats:    0 → 18,506  ✅
injury_reports: NULL → 30,085  ✅
monster_p_is:   NULL → 76  ✅
orphaned matches: 11,485 → 0  ✅ (FK посилання тепер валідні)
```

Після цього всі 3 генератори запрацювали:
- WS Gap: 29 → 33 (+4)
- Monster: 21 → 23 (+2, Telegram broadcast ОК)
- Aquamarine: 7 → 7 (0 — найконсервативніший, нічого не пройшло фільтр)

## Урок

**Schema drift можна виявляти проактивно.** Ідея: додати scheduler-task раз на годину / при старті, який робить `SELECT 1 FROM match_stats LIMIT 1` / `EXISTS injury_reports` і логує попередження при розбіжності з моделями. Поки що — ручна перевірка через `\d` при кожному deploy.

## Related

- [VPS SStats Proxy](vps-sstats-proxy.md)
- [Project Status](project-status.md)
- [Bug Audit 2026-04-22](bug-audit-2026-04-22.md)
