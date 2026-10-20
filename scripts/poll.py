"""Run the bot locally with long polling instead of the Vercel webhook (for live testing).

    TELEGRAM_BOT_TOKEN=... OPEN_ROUTER_KEY=... python scripts/poll.py --memory --echo

--memory  keep messages in memory instead of Supabase
--echo    reply with the verdict in the chat the message came from
Polling and a webhook can't coexist: this removes the webhook while it runs.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import replace  # noqa: E402

from bot.config import Settings  # noqa: E402
from bot.service import Bot  # noqa: E402
from bot.storage import MemoryStorage, Storage  # noqa: E402
from bot.telegram import Telegram  # noqa: E402
from classifier.dataset import load_cases  # noqa: E402
from classifier.fewshot import FewShotSelector  # noqa: E402
from classifier.jev import Jev, JevConfig  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--memory", action="store_true")
parser.add_argument("--echo", action="store_true")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
settings = Settings.from_env()
if args.echo:
    settings = replace(settings, echo_reports=True)
telegram = Telegram(settings.telegram_token)
telegram.client.timeout = 60
storage = MemoryStorage() if args.memory else Storage(settings.supabase_url, settings.supabase_key)
bot = Bot(settings, telegram, storage, Jev(JevConfig.from_env(), selector=FewShotSelector(load_cases())))

telegram.call("deleteWebhook")
me = telegram.call("getMe")
logging.info("polling as @%s (memory=%s, echo=%s)", me["username"], args.memory, settings.echo_reports)
offset = 0
while True:
    for update in telegram.call("getUpdates", offset=offset, timeout=50, allowed_updates=["message", "edited_message"]):
        offset = update["update_id"] + 1
        try:
            status = bot.handle(update)
        except Exception:
            logging.exception("update %s failed", update["update_id"])
            continue
        logging.info("update %s: %s", update["update_id"], status)
