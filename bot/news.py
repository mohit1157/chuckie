import logging
from .config import AppConfig

LOG = logging.getLogger("bot.news")

class NewsFilter:
    '''
    Pluggable news / economic calendar filter.

    Default provider is a stub that never blocks.
    Replace provider="stub" with your own implementation.
    '''
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def block_new_entries(self) -> bool:
        if self.cfg.news_filter.provider == "stub":
            return False
        LOG.warning("Unknown news provider '%s' - not blocking.", self.cfg.news_filter.provider)
        return False
