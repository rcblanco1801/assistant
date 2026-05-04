import os, json, time, uuid, zlib, asyncio, uvicorn
from typing import AsyncIterator, Dict, Any, List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from modules.assistant import Assistant, ModelName
from fastapi.encoders import jsonable_encoder
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory


POOL_SIZE = 6
MODEL = ModelName.GRANITE_MICRO
TEMPERATURE = 0.0
TOP_K = 3
MAX_MEMORY_LENGTH = 50 # Máximo número de mensajes que se pasan como historial al LLM

class AssistantPool:
    def __init__(self, size: int):
        self._size = size
        self._assistants: List[Assistant] = [
            Assistant(model_name=MODEL, temperature=TEMPERATURE, k=TOP_K) 
            for _ in range(size)
        ]
        self._locks: List[asyncio.Lock] = [asyncio.Lock() for _ in range(size)]
        self._memories: Dict[str, ConversationBufferMemory] = {}

    def _pick_slot(self, user_id: str) -> int:
        # hashing consistente por usuario -> minimiza "thrash" de memoria de conversación
        return zlib.crc32(user_id.encode("utf-8")) % self._size

    def get(self, user_id: str):
        idx = self._pick_slot(user_id)
        return idx, self._assistants[idx], self._locks[idx]
    
    def forget(self, user_id: str):
        self._memories.pop(user_id, None)

    def get_memory(self, user_id: str) -> ConversationBufferMemory:
        if user_id not in self._memories:
            self._memories[user_id] = ConversationBufferMemory(
                chat_memory=ChatMessageHistory(),
                memory_key="chat_history",
                input_key="question",
                return_messages=True,
            )
        return self._memories[user_id]

pool = AssistantPool(POOL_SIZE)
app = FastAPI(title="llama.cpp-compatible shim for Openfire")

def _now_unix() -> int:
    return int(time.time())

def _chat_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex

def _last_user_content(messages: List[Dict[str, Any]]) -> str:
    # usamos la memoria interna del Assistant por user_id; aun así tomamos el último user
    for m in reversed(messages or []):
        if (m.get("role") == "user") and m.get("content"):
            return str(m["content"])
    return ""

def _sse(data: Dict[str, Any]) -> str:
    # pasar de Dict a json
    payload = jsonable_encoder(data)    
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/v1/models")
async def list_models():
    # respuesta mínima OpenAI-like
    return {
        "object": "list",
        "data": [
            {
                "id": str(MODEL),
                "object": "model",
                "created": _now_unix(),
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    """
    Espera cuerpo estilo OpenAI:
    {
      "model": "...",
      "messages": [{"role":"system"|"user"|"assistant","content":"..."}],
      "stream": true|false,
      "max_tokens": ..., "temperature": ..., "top_p": ..., "stop": [...]
    }
    """
    try:
        body = await req.json()
    except Exception:
        raise HTTPException(400, "JSON inválido")

    messages = body.get("messages", [])
    stream = bool(body.get("stream", False))
    model_name = MODEL
    user_id = req.headers.get("X-User") or body.get("user") or "anon"

    # prompt actual (la memoria por user_id la gestiona Assistant)
    prompt = _last_user_content(messages)
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(400, "Faltan 'messages' con al menos un mensaje de 'user'")

    idx, assistant, lock = pool.get(user_id)
    created = _now_unix()
    chat_id = _chat_id()

    memory = pool.get_memory(user_id)
    history = memory.load_memory_variables({}).get("chat_history", [])

    # Borramos memoria cuando el historial supere el máximo de mensajes
    if MAX_MEMORY_LENGTH <= len(history):
        pool.forget(user_id)

    if not stream:
        # Respuesta completa no streaming
        text_chunks: List[str] = []
        async with lock:
            async for tok in assistant.ask_stream(prompt, history=history):
                text_chunks.append(tok)
        full_text = "".join(text_chunks)
        memory.save_context({"question": prompt}, {"response": full_text})
        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_text},
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    # Streaming SSE tipo OpenAI
    async def gen() -> AsyncIterator[str]:
        # 1ª delta con "role"
        yield _sse({
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        })
        async with lock:
            text_chunks: List[str] = []
            async for tok in assistant.ask_stream(prompt, history=history):
                yield _sse({
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": _now_unix(),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}]
                })
                text_chunks.append(tok)
        full_text = "".join(text_chunks)
        memory.save_context({"question": prompt}, {"response": full_text})
        # chunk final
        yield _sse({
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": _now_unix(),
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        })
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

@app.post("/completion")
async def legacy_completion(req: Request):
    """
    Compatible con llama.cpp /completion:
    - prompt: str | tokens | lista mixta
    - stream: bool (SSE cuando True)
    - n_predict, temperature, top_k, top_p, stop, ...
    """
    try:
        body = await req.json()
    except Exception:
        raise HTTPException(400, "JSON inválido")

    prompt = body.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(400, "Falta 'prompt'")

    stream = bool(body.get("stream", False))
    user_id = req.headers.get("X-User") or "anon"
    model_name = str(MODEL)
    idx, assistant, lock = pool.get(user_id)

    memory = pool.get_memory(user_id)
    history = memory.load_memory_variables({}).get("chat_history", [])

    # Borramos memoria cuando el historial supere el máximo de mensajes
    if MAX_MEMORY_LENGTH <= len(history):
        pool.forget(user_id)

    # (Opcional) lee parámetros de sampling si luego los pasas a tu motor
    n_predict = body.get("n_predict")
    temperature = body.get("temperature")
    top_k = body.get("top_k")
    top_p = body.get("top_p")
    stop_words = body.get("stop", [])

    if not stream:
        text_chunks: List[str] = []
        async with lock:
            text_chunks: List[str] = []
            async for tok in assistant.ask_stream(prompt, history=history):
                text_chunks.append(tok)
        full_text = "".join(text_chunks)
        memory.save_context({"question": prompt}, {"response": full_text})
        return {
            "id": "cmpl-" + uuid.uuid4().hex,
            "content": "".join(text_chunks),
            "stop": True,
            "model": model_name,
            # (Opcional) eco mínimo de settings para parecerte más a llama.cpp
            "generation_settings": {
                "n_predict": n_predict, "temperature": temperature,
                "top_k": top_k, "top_p": top_p, "stop": stop_words
            }
        }

    # --- Streaming SSE (como espera llama.cpp) ---
    async def gen_sse() -> AsyncIterator[str]:
        # Cabecera: cada chunk debe ser "data: {json}\n\n"
        async with lock:
            text_chunks: List[str] = []
            async for tok in assistant.ask_stream(prompt, history=history):
                ev = {"content": tok, "stop": False}
                yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
                text_chunks.append(tok)
        full_text = "".join(text_chunks)
        memory.save_context({"question": prompt}, {"response": full_text})
        # Evento final de parada
        yield "data: " + json.dumps({"stop": True}, ensure_ascii=False) + "\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        # "Connection": "keep-alive"  # Nginx lo gestionará
    }
    return StreamingResponse(gen_sse(), media_type="text/event-stream", headers=headers)

@app.post("/v1/clear")
async def clear(req: Request):
    user_id = req.headers.get("X-User", "anon")
    pool.forget(user_id)
    return JSONResponse({"ok": True})

# opcional: raíz informativa
@app.get("/")
async def root():
    return PlainTextResponse("llama.cpp-compatible server: /v1/chat/completions, /completion, /v1/models, /health")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)