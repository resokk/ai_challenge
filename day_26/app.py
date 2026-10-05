"""Минимальный веб-чат с локальной моделью Ollama.

Только стандартная библиотека. История диалогов живёт в памяти процесса
и пропадает при перезапуске.

    python app.py [--model qwen2.5-coder:7b] [--port 8000] [--num-ctx 32768]
"""

import argparse
import json
import threading
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
SYSTEM_PROMPT = "Ты полезный ассистент. Отвечай на языке пользователя."
MAX_MESSAGE_CHARS = 20_000
INDEX_HTML = (Path(__file__).parent / "index.html").read_bytes()


class Sessions:
    """История сообщений по id сессии, только в памяти."""

    def __init__(self):
        self._lock = threading.Lock()
        self._history: dict[str, list[dict]] = {}

    def get(self, sid: str) -> list[dict]:
        with self._lock:
            return list(self._history.get(sid, []))

    def append_turn(self, sid: str, user: str, assistant: str) -> None:
        with self._lock:
            self._history.setdefault(sid, []).extend(
                [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
            )

    def reset(self, sid: str) -> None:
        with self._lock:
            self._history.pop(sid, None)


def stream_ollama(model: str, num_ctx: int, messages: list[dict]):
    """Отдаёт куски ответа модели по мере генерации."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        "stream": True,
        "options": {"num_ctx": num_ctx},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            chunk = json.loads(line)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            yield chunk.get("message", {}).get("content", "")
            if chunk.get("done"):
                return


class Handler(BaseHTTPRequestHandler):
    sessions: Sessions
    model: str
    num_ctx: int

    def do_GET(self):
        if self.path == "/":
            self._send(200, INDEX_HTML, "text/html; charset=utf-8")
        elif self.path.startswith("/api/history?sid="):
            sid = self.path.split("=", 1)[1]
            self._send_json(200, {"messages": self.sessions.get(sid), "model": self.model})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            sid = str(uuid.UUID(data["sid"]))
        except (ValueError, KeyError, TypeError):
            return self._send_json(400, {"error": "bad request"})

        if self.path == "/api/reset":
            self.sessions.reset(sid)
            return self._send_json(200, {"ok": True})
        if self.path != "/api/chat":
            return self._send_json(404, {"error": "not found"})

        text = data.get("message")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_MESSAGE_CHARS:
            return self._send_json(400, {"error": f"сообщение пустое или длиннее {MAX_MESSAGE_CHARS} символов"})

        messages = self.sessions.get(sid) + [{"role": "user", "content": text}]
        # Тело без Content-Length: ответ идёт кусками до закрытия соединения.
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        answer = []
        try:
            for piece in stream_ollama(self.model, self.num_ctx, messages):
                answer.append(piece)
                self.wfile.write(piece.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return  # клиент ушёл: незаконченный ход в историю не пишем
        except (urllib.error.URLError, RuntimeError, json.JSONDecodeError) as e:
            self.wfile.write(f"\n\n[ошибка Ollama: {e}]".encode())
            return
        self.sessions.append_turn(sid, text, "".join(answer))

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")


def main():
    p = argparse.ArgumentParser(description="Веб-чат с локальной моделью Ollama")
    p.add_argument("--model", default="qwen2.5-coder:7b")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--num-ctx", type=int, default=32768)
    args = p.parse_args()

    Handler.sessions = Sessions()
    Handler.model = args.model
    Handler.num_ctx = args.num_ctx
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Чат: http://127.0.0.1:{args.port}  (модель {args.model}, контекст {args.num_ctx})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
