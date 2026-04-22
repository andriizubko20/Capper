import time
import httpx
from loguru import logger
from config.settings import settings

NETWORK_RETRY_DELAY = 60  # секунд між спробами при мережевій помилці
NETWORK_MAX_RETRIES = 10  # ~10 хв максимум, далі таск завершується з помилкою


class SStatsAPIError(Exception):
    pass


class SStatsClient:
    def __init__(self):
        self._client = httpx.Client(
            base_url=settings.sstats_api_host,
            headers={"X-API-KEY": settings.sstats_api_key},
            timeout=30.0,
        )

    def get(self, endpoint: str, params: dict | None = None) -> dict | list:
        attempt = 0
        while True:
            try:
                response = self._client.get(endpoint, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 429:
                    attempt += 1
                    wait = min(5 * attempt, 30)  # 5s, 10s, 15s... max 30s
                    logger.warning(f"Rate limit 429 {endpoint} — повтор через {wait}s (спроба {attempt})")
                    time.sleep(wait)
                else:
                    logger.error(f"SStats API error: {status} {endpoint}")
                    raise SStatsAPIError(f"HTTP {status}: {endpoint}") from e
            except httpx.RequestError as e:
                attempt += 1
                if attempt >= NETWORK_MAX_RETRIES:
                    logger.error(f"SStats connection failed after {NETWORK_MAX_RETRIES} retries: {endpoint}")
                    raise SStatsAPIError(f"Connection failed after {NETWORK_MAX_RETRIES} retries: {endpoint}") from e
                logger.warning(f"SStats connection error: {endpoint} — повтор через {NETWORK_RETRY_DELAY}s (спроба {attempt}/{NETWORK_MAX_RETRIES})")
                time.sleep(NETWORK_RETRY_DELAY)

    def post(self, endpoint: str, body: dict) -> dict | list:
        try:
            response = self._client.post(endpoint, json=body)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"SStats API error: {e.response.status_code} {endpoint}")
            raise SStatsAPIError(f"HTTP {e.response.status_code}: {endpoint}") from e
        except httpx.RequestError as e:
            logger.error(f"SStats connection error: {endpoint} — {e}")
            raise SStatsAPIError(f"Connection error: {endpoint}") from e

        return response.json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
