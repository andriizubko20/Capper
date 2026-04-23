# VPS ↔ SStats Proxy Workaround

**Category:** Tools & Tech
**Created:** 2026-04-22
**Updated:** 2026-04-22

Обхід мережевого обриву DigitalOcean ↔ api.sstats.net через локальний проксі.

## Проблема

VPS DigitalOcean (165.227.164.220, NYC) не отримує повну відповідь від `api.sstats.net` (213.171.21.1, Fasthosts UK). Сервер надсилає частину chunked-response, потім TCP-потік обривається — фінальний chunk-термінатор `0\r\n\r\n` не доходить.

**Симптоми:**
- `curl` з VPS → таймаут після 3+ хвилин, отримано лише 14,751 з 363,667 байт (4%)
- `httpx`, `urllib`, `aiohttp`, `http.client` — всі падають з `ReadTimeout`
- Raw socket отримує 15 KB, далі recv блокується
- З локальної машини той самий запит — 2 сек, 363 KB, валідний JSON ✅

**Не допомогло:**
- Додавання `SSTATS_API_KEY` у `.env` (була відсутня — це був окремий баг)
- iptables TCPMSS clamp (на POSTROUTING + FORWARD)
- HTTP/1.1 замість HTTP/2
- `verify=False` (SSL)
- Встановлення `curl` у Docker-контейнер (той самий результат що й Python)

**Діагноз:** network path broken between DigitalOcean/NYC and Fasthosts/UK. Вірогідно peering або PMTU blackhole на проміжному маршрутизаторі. Не виправляється на боці клієнта.

## Поточне рішення (тимчасове)

**Ланцюг:** Docker container → 172.18.0.1:8766 → socat → 127.0.0.1:8765 → SSH reverse tunnel → Mac → Python proxy → `https://api.sstats.net`

### Компоненти

**1. Python proxy на Mac — `/tmp/sstats_proxy.py`**

```python
import http.server, urllib.request, ssl, sys

TARGET = "https://api.sstats.net"

class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            req = urllib.request.Request(TARGET + self.path)
            for h, v in self.headers.items():
                if h.lower() in ("host", "connection", "content-length"):
                    continue
                req.add_header(h, v)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                body = resp.read()
                self.send_response(resp.status)
                for h, v in resp.headers.items():
                    if h.lower() in ("transfer-encoding", "connection", "content-encoding"):
                        continue
                    self.send_header(h, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f"proxy error: {e}".encode())

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    http.server.ThreadingHTTPServer(("127.0.0.1", port), ProxyHandler).serve_forever()
```

Запуск: `nohup python3 /tmp/sstats_proxy.py 8765 > /tmp/sstats_proxy.log 2>&1 &`

**2. SSH reverse tunnel** (Mac → VPS)
```
ssh -f -N -R 8765:localhost:8765 root@165.227.164.220
```

**3. socat forwarder на VPS** (щоб Docker-контейнер міг достукатися)
```
nohup socat TCP-LISTEN:8766,fork,reuseaddr,bind=172.18.0.1 TCP:127.0.0.1:8765 > /tmp/socat.log 2>&1 &
```

**4. UFW rule**
```
ufw allow from 172.18.0.0/16 to 172.18.0.1 port 8766 proto tcp comment 'capper-proxy-tunnel'
```

**5. `.env` на VPS**
```
SSTATS_API_HOST=http://172.18.0.1:8766
```

## Автоматизація (2026-04-22)

Всі 3 компоненти ланцюга тепер переживають crash / sleep / reboot:

### Mac (launchd)

Файли:
- `~/.capper/sstats_proxy.py` — постійне розташування скрипта
- `~/Library/LaunchAgents/com.capper.sstats-proxy.plist` — Python proxy на порту 8765
- `~/Library/LaunchAgents/com.capper.sstats-tunnel.plist` — SSH reverse tunnel

Обидва з `RunAtLoad=true`, `KeepAlive=true`, `ThrottleInterval=5-10s`. Логи в `~/Library/Logs/capper/`.

SSH tunnel з параметрами `ServerAliveInterval=30`, `ServerAliveCountMax=2`, `ExitOnForwardFailure=yes` — при обриві (або sleep/wake) ssh виходить, launchd одразу перезапускає.

Управління:
```bash
launchctl list | grep capper
launchctl unload ~/Library/LaunchAgents/com.capper.sstats-proxy.plist
launchctl load ~/Library/LaunchAgents/com.capper.sstats-proxy.plist
```

### VPS (systemd)

`/etc/systemd/system/capper-proxy-forwarder.service` — socat TCP forwarder `172.18.0.1:8766 → 127.0.0.1:8765` з `Restart=always`, `RestartSec=5`, enabled для multi-user target.

```bash
systemctl status capper-proxy-forwarder
journalctl -u capper-proxy-forwarder -f
```

## Обмеження

- **Mac має бути увімкнений** — це єдиний single-point-of-failure.
- Все інше переживає rebootу / crash / sleep.
- 502 помилки при тайм-ауті 30s на Mac-проксі (деякі fixtures повертаються повільно) — acceptable, такі fixtures скіпаються і retry пізніше.

## Cloudflare Worker не працює

Тестовано 2026-04-22: `aqua.andrii-zubko20.workers.dev` отримує ті самі truncated 14,751 байт за 77 секунд. Отже **api.sstats.net обмежує datacenter-IP-діапазони в цілому**, не лише DigitalOcean NYC. Cloudflare Workers, Hetzner, GCP теж вірогідно не вирішать проблему.

## 100% uptime — TODO

- **SStats support** — попросити whitelist VPS IP `165.227.164.220`. Правильний шлях, бо ми клієнти API. 1-7 днів.
- **Raspberry Pi вдома** — home-ISP = працює як Mac, always-on, ~$50 одноразово.
- **Перенести scheduler на home-machine** — перемістити весь Docker stack, не тільки proxy.

## Related

- [SStats API](sstats-api.md)
- [Capper Overview](capper-overview.md)
- [DB Restore 2026-04-22](db-restore-2026-04-22.md)
