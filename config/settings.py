from pydantic_settings import BaseSettings

# Canonical model_version values that count toward bankroll and stats.
# All other versions (base non-Kelly, etc.) are parallel runs excluded from UI.
CANONICAL_VERSIONS: frozenset[str] = frozenset({
    "ws_gap_kelly_v1",
    "monster_v1_kelly",
    "aquamarine_v1_kelly",
})


class Settings(BaseSettings):
    telegram_bot_token: str = ""
    sstats_api_key: str = ""
    sstats_api_host: str = "https://api.sstats.net"
    api_football_key: str = ""         # API-Football v3 (free tier 100 req/day)
    database_url: str = "postgresql://capper:capper@localhost:5432/capper"
    env: str = "development"

    # Betting
    bankroll: float = 1000.0
    picks_hours_before: int = 5        # фінальний пік за N годин до старту матчу
    picks_hours_before_late: float = 1.5  # late-pass pick after lineups confirmed
    early_picks_days_ahead: int = 4    # ранній пік — матчі до N днів вперед

    # Mini App URL
    miniapp_url: str = "https://aqua-traders.online"

    # Access control: comma-separated Telegram user IDs, empty = open for all
    allowed_telegram_ids: str = ""

    @property
    def allowed_ids_set(self) -> set[int]:
        if not self.allowed_telegram_ids.strip():
            return set()
        return {int(x.strip()) for x in self.allowed_telegram_ids.split(",") if x.strip()}

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


settings = Settings()
