from openai import AsyncOpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

SYSTEM_PROMPT = "You are a helpful assistant chatting with a user over Telegram. Keep replies concise."

client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


async def get_reply(history: list[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
    response = await client.chat.completions.create(model=DEEPSEEK_MODEL, messages=messages)
    return response.choices[0].message.content
