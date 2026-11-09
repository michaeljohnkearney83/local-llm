"""Exercise 3: talk to a local Ollama model from Python.

No extra packages needed: this uses only Python's standard library to send HTTP requests
to Ollama's local API at http://127.0.0.1:11434.

Usage:
    python chat.py                          # interactive chat with llm-tutor
    python chat.py --model qwen3:8b         # chat with a different model
    python chat.py --ask "What is a token?" # ask one question and exit

Key ideas this script shows:
  1. An LLM is just a web service on your PC: you send JSON, it sends text back.
  2. Streaming: the reply arrives a few tokens at a time, so you can print it as it is written.
  3. The model has no memory. "Memory" is us re-sending the whole conversation every turn.
"""
import argparse
import json
import time
import urllib.error
import urllib.request

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


def chat(model, messages):
    """Send the conversation so far to Ollama, print the reply as it streams in, and return it."""
    request_body = {
        "model": model,
        "messages": messages,  # the WHOLE conversation, every time (idea 3)
        "stream": True,        # get the reply piece by piece (idea 2)
        "think": False,        # skip Qwen3's slow step-by-step reasoning on the CPU
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    reply = ""
    stats = {}
    with urllib.request.urlopen(request, timeout=600) as response:
        # Ollama streams one small JSON object per line, e.g. {"message": {"content": "Hel"}, "done": false}
        for line in response:
            chunk = json.loads(line)
            piece = chunk.get("message", {}).get("content", "")
            print(piece, end="", flush=True)
            reply += piece
            if chunk.get("done"):
                stats = chunk  # the final chunk carries timing information
    print()

    # Durations are in nanoseconds.
    tokens = stats.get("eval_count", 0)
    seconds = stats.get("eval_duration", 1) / 1e9
    prompt_tokens = stats.get("prompt_eval_count", 0)
    print(f"  [{tokens} tokens at {tokens / seconds:.1f} tokens/sec | "
          f"model read {prompt_tokens} tokens of conversation]")
    return reply


def main():
    parser = argparse.ArgumentParser(description="Chat with a local Ollama model.")
    parser.add_argument("--model", default="llm-tutor", help="model name (default: llm-tutor)")
    parser.add_argument("--ask", help="ask a single question and exit")
    args = parser.parse_args()

    messages = []  # the conversation history: a list of {"role": ..., "content": ...}

    try:
        if args.ask:
            messages.append({"role": "user", "content": args.ask})
            chat(args.model, messages)
            return

        print(f"Chatting with {args.model}. Type 'quit' to exit, 'reset' to forget the conversation.\n")
        while True:
            question = input("You: ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "bye"):
                break
            if question.lower() == "reset":
                messages.clear()
                print("(Conversation forgotten. The model now knows nothing about what you said.)\n")
                continue

            messages.append({"role": "user", "content": question})
            print(f"{args.model}: ", end="")
            reply = chat(args.model, messages)
            messages.append({"role": "assistant", "content": reply})  # remember its answer too
            print()
    except urllib.error.URLError:
        print("Can't reach Ollama at http://127.0.0.1:11434. Is it running? Open Ollama from the Start menu.")
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
