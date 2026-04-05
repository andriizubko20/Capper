from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    telegram_bot_token: str = ""
    sstats_api_key: str = ""
    sstats_api_host: str = "https://api.sstats.net"
    database_url: str = "postgresql://capper:capper@localhost:5432/capper"
    env: str = "development"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


settings = Settings()
