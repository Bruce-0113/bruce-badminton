"""Gradio 聊天介面，連到本機的 mlx_lm.server。

    mlx_lm.server --model out/coach-m0-q4 --port 8080
    python -m coach.app
"""

import gradio as gr
from openai import OpenAI

from coach.generate import COACH_SYSTEM

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")


def respond(message, history):
    messages = [{"role": "system", "content": COACH_SYSTEM}]
    messages += [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": message})
    stream = client.chat.completions.create(
        model="default_model",
        messages=messages,
        max_tokens=1024,
        temperature=0.6,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
            yield text


def main():
    gr.ChatInterface(
        respond,
        type="messages",
        title="🏸 羽球教練（M0）",
        examples=["我殺球總是沒力，該怎麼練？", "雙打發球的高度限制是多少？", "打完球膝蓋外側會痛，要怎麼辦？"],
    ).launch()


if __name__ == "__main__":
    main()
