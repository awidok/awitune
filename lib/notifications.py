"""Notification helpers."""


def send_telegram_notification(cfg, message: str, parse_mode: str = "HTML"):
    """Send a notification to Telegram if configured."""
    if cfg is None:
        return

    bot_token = cfg.telegram_bot_token
    chat_id = cfg.telegram_chat_id

    if not bot_token or not chat_id:
        return

    try:
        import requests

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": parse_mode,
        }
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            print(f"[telegram] Failed to send notification: {response.status_code} {response.text}")
    except Exception as e:
        print(f"[telegram] Error sending notification: {e}")
