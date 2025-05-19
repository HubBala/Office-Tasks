import requests

def query_rasa_bot(message):
    """Send message to Rasa REST webhook and get response text."""
    try:
        response = requests.post(
            "http://localhost:5005/webhooks/rest/webhook",
            json={"sender": "user", "message": message},
            timeout=5
        )
        response.raise_for_status()
        bot_messages = response.json()
        # Return concatenated bot texts or a default message
        return " ".join([msg.get("text", "") for msg in bot_messages]) or "No response from bot."
    except Exception as e:
        return f"Error contacting Rasa server: {e}"
