"""Point the Telegram bot at the deployed webhook.

    TELEGRAM_BOT_TOKEN=... TELEGRAM_WEBHOOK_SECRET=... python scripts/set_webhook.py https://<app>.vercel.app
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot.telegram import Telegram  # noqa: E402

if len(sys.argv) != 2:
    sys.exit(__doc__)
token, secret = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_WEBHOOK_SECRET"]
telegram = Telegram(token)
telegram.set_webhook(sys.argv[1].rstrip("/") + "/api/webhook", secret)
print(telegram.call("getWebhookInfo"))
