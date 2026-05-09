import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))


def call_groq(system_prompt: str, messages: list[dict]) -> str:
    """
    Calls Groq (llama-3.3-70b-)
    (Raises exception on API failure — handled in recommendation_service)
    """
    groq_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        groq_messages.append({"role": msg["role"], "content": msg["content"]})

    response = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=groq_messages,
        temperature=0.2,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content