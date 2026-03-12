#!/usr/bin/env python3
"""Minimal chat with any HuggingFace model. No server, no hooks, just chat.
Usage: python chat.py [model_name_or_path]
"""
import sys, os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings; warnings.filterwarnings("ignore")
import logging
for n in ["huggingface_hub","transformers","accelerate","torch"]:
    logging.getLogger(n).setLevel(logging.CRITICAL)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"
print(f"Loading {model_name}...")
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
print("Ready.\n")

history = []
while True:
    try:
        q = input("> ")
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not q.strip():
        continue
    if q.strip() in ("/quit", "/exit", "quit", "exit"):
        break
    if q.strip() == "/clear":
        history = []
        print("cleared")
        continue
    history.append({"role": "user", "content": q})
    try:
        text = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        text = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    Thread(target=model.generate, kwargs={**inputs, "max_new_tokens": 512, "do_sample": True, "temperature": 0.7, "top_p": 0.9, "streamer": streamer}).start()
    resp = ""
    for t in streamer:
        sys.stdout.write(t)
        sys.stdout.flush()
        resp += t
    print()
    history.append({"role": "assistant", "content": resp})
