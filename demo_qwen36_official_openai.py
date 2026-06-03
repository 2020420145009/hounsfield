#!/usr/bin/env python3
"""
Official Qwen3.6 Chat Completions demo, localized for a self-hosted OpenAI-compatible endpoint.
Default behavior: start a local vLLM server for this inference only, call it once, then stop it.
Set QWEN_USE_EXISTING_SERVER=1 to reuse an already running endpoint instead.
"""

import os

from openai import OpenAI

from qwen36_local_runtime import managed_server


messages = [
    {"role": "user", "content": 'Type "I love Qwen3.6" backwards'},
]

use_existing_server = os.environ.get("QWEN_USE_EXISTING_SERVER", "0") == "1"
with managed_server(start_if_needed=not use_existing_server) as (base_url, served_model_name, _process):
    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", base_url),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )

    chat_response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", served_model_name),
        messages=messages,
        max_tokens=81920,
        temperature=1.0,
        top_p=0.95,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
        },
    )

    print(chat_response)
