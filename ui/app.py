"""Streamlit chat client for the API.

Sends the API key the server now requires, and keeps the API URL fixed to
configuration: the old sidebar let anyone point this server-side client at
any URL, which made the UI container a request proxy.
"""

import asyncio
import json
import os

import aiohttp
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL") or f"http://localhost:{os.getenv('APP_PORT', '8058')}"
API_KEY = os.getenv("API_KEY", "")
USER_ID = os.getenv("UI_USER_ID", "ui")
HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def check_health() -> dict | None:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.json() if resp.status_code in (200, 503) else None
    except Exception:
        return None


async def stream_chat(message: str) -> None:
    payload = {
        "message": message,
        "session_id": st.session_state.session_id,
        "user_id": USER_ID,
        "search_type": "hybrid",
    }
    box = st.empty()
    full_response = ""
    async with aiohttp.ClientSession(headers=HEADERS) as session:  # noqa: SIM117
        async with session.post(f"{API_URL}/chat/stream", json=payload) as resp:
            if resp.status != 200:
                detail = (await resp.json()).get("detail", resp.reason)
                box.error(f"API error {resp.status}: {detail}")
                return
            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                kind = data.get("type")
                if kind == "session":
                    st.session_state.session_id = data.get("session_id")
                elif kind == "text":
                    full_response += data.get("content", "")
                    box.write(full_response)
                elif kind == "tools":
                    for tool in data.get("tools", []):
                        full_response += (
                            f"\n\n[Tool: {tool.get('tool_name', '')}] args: {tool.get('args', {})}"
                        )
                    box.write(full_response)
                elif kind == "error":
                    box.error(data.get("content", "The agent failed"))
                    return
                elif kind == "end":
                    break
    box.write(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})


def run_async(message: str) -> None:
    asyncio.run(stream_chat(message))


with st.sidebar:
    st.header("Settings")
    st.caption(f"API: {API_URL}")
    if not API_KEY:
        st.warning("API_KEY is not set; requests will be rejected.")
    if st.button("Check health"):
        health = check_health()
        if health and health.get("status") == "healthy":
            st.success("API is healthy")
        elif health:
            st.error(f"API is {health.get('status')}: database={health.get('database')}")
        else:
            st.error("API not reachable")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

st.title("Agentic RAG")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        run_async(prompt)
