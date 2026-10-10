import os

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
# Any Hugging Face chat model whose template supports tools; downloaded on first use.
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LOCAL_LLM_MAX_NEW_TOKENS = int(os.environ.get("LOCAL_LLM_MAX_NEW_TOKENS", "1024"))
