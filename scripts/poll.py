"""Run the bot locally with long polling instead of the Vercel webhook (for live testing).

    TELEGRAM_BOT_TOKEN=... OPEN_ROUTER_KEY=... CLIENT_USERNAMES=... REPORT_TO=... python scripts/poll.py --memory

--memory  keep messages in memory instead of Supabase
Polling and a webhook can't coexist: this removes the webhook while it runs.
"""

import argparse
import logging
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot.config import Settings  # noqa: E402
from bot.service import Bot  # noqa: E402
from bot.storage import MemoryStorage, Storage  # noqa: E402
from bot.telegram import Telegram, TelegramError  # noqa: E402
from classifier.dataset import load_cases  # noqa: E402
from classifier.fewshot import FewShotSelector  # noqa: E402
from classifier.jev import Jev, JevConfig  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--memory", action="store_true")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
settings = Settings.from_env()
telegram = Telegram(settings.telegram_token)
telegram.client.timeout = httpx.Timeout(40)
storage = MemoryStorage() if args.memory else Storage(settings.supabase_url, settings.supabase_key)
bot = Bot(settings, telegram, storage, Jev(JevConfig.from_env(), selector=FewShotSelector(load_cases())))

telegram.call("deleteWebhook")
me = telegram.call("getMe")
logging.info("polling as @%s (memory=%s, clients=%s, report_to=%s)", me["username"], args.memory,
             sorted(settings.client_usernames), sorted(settings.report_usernames))
offset = 0
while True:
    try:
        # Short long-poll: proxies tend to cut idle connections after ~30s.
        updates = telegram.call("getUpdates", offset=offset, timeout=20, allowed_updates=["message", "edited_message"])
    except (httpx.HTTPError, TelegramError) as exc:
        logging.warning("getUpdates failed, retrying: %s", exc)
        time.sleep(3)
        continue
    for update in updates:
        offset = update["update_id"] + 1
        try:
            status = bot.handle(update)
        except Exception:
            logging.exception("update %s failed", update["update_id"])
            continue
        logging.info("update %s: %s", update["update_id"], status)
