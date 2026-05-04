import time, os, re

from langchain_core.language_models import BaseChatModel
from langchain.docstore.document import Document
from langchain_ollama import ChatOllama
from modules.data_loader import DataLoader
from modules.data_retriever import DataRetriever
from modules.image_describer import ImageDescriber

from enum import Enum
from typing import List, AsyncIterator, Any
from datetime import datetime
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, BaseMessage
)


def extract_img_urls(doc: str) -> list[str]:
    pattern = re.compile(r"<\s*img_url\s*>(.*?)</\s*img_url\s*>",
                        flags=re.IGNORECASE | re.DOTALL)
    return [m.strip() for m in pattern.findall(doc)]

def build_user_message(query: str, imgs: List[str], 
                       describer: ImageDescriber = None) -> HumanMessage:
    parts = [{"type": "text", "text": query}]
    if describer:
        parts.extend([{"type": "text", "text": describer.describe(p)} for p in imgs])

    return HumanMessage(content=parts)

def build_context_message(prompt: str,
                          docs: List[str],
                          imgs: List[str] = []) -> SystemMessage | None:
    if not docs and not imgs:
        return None

    parts = [{"type": "text", "text": prompt}]     
    parts.extend([{"type": "text", "text": t} for t in docs])

    return SystemMessage(content=parts)

def build_system_message(prompt: str) -> SystemMessage:
    return SystemMessage(
        content=prompt
    )

class ModelName(Enum):
    LLAMA=1
    QWEN=2
    GEMMA=3
    GRANITE_TINY=4
    GRANITE_MICRO=5
    GRANITE_NANO=6
    DEEPSEEK=7
    GPT=8

def get_llm(model_name: ModelName, temperature: float) -> BaseChatModel:
    if model_name == ModelName.LLAMA:
        return ChatOllama(model="llama3.2:3b", temperature=temperature)
    elif model_name == ModelName.GEMMA:
        return ChatOllama(model="gemma3n:e4b", temperature=temperature)
    elif model_name == ModelName.QWEN:
        return ChatOllama(model="qwen3:4b-instruct", temperature=temperature)
    elif model_name == ModelName.GRANITE_TINY:
        return ChatOllama(model="granite4:tiny-h", temperature=temperature)
    elif model_name == ModelName.GRANITE_MICRO:
        return ChatOllama(model="granite4:micro-h", temperature=temperature)
    elif model_name == ModelName.GRANITE_NANO:
        return ChatOllama(model="granite4:1b-h", temperature=temperature)
    elif model_name == ModelName.DEEPSEEK:
        return ChatOllama(model="deepseek-r1:8b", temperature=temperature)
    elif model_name == ModelName.GPT:
        return ChatOllama(model="gpt-oss:20b", temperature=temperature)
    
class Assistant:
    def __init__(self, model_name = ModelName.LLAMA, temperature: float = 0.05, k: int = 3):
        self._benchmark_dir = "benchmark"
        self._loader = DataLoader("./data", "./figures")
        self._retriever = DataRetriever(self._loader, "./storage", top_k=k, assistant=True)
        self._model_name = model_name
        self._top_k = k
        # self._multimodal = self._model_name != ModelName.LLAMA
        self._llm = get_llm(model_name, temperature)
        self._imgs_dir = "query_imgs"

        # if not self._multimodal:
        #     self._img_describer = ImageDescriber()
        # else:
        #     self._img_describer = None

        self._system_prompt = (
            "Eres el asistente de la sede del Ayuntamiento de Benalmádena, España. "
            "Tu trabajo es asistir al funcionariado dentro de la red interna para resolver dudas técnicas acerca de las herramientas electrónicas varias, trámites administrativos y demás. "
            "Responde en español e, importante, si no estás seguro de la respuesta a una pregunta di que no la conoces y remite al usuario al departamento de informática o al Moodle del ayuntamiento, https://moodle.benalmadena.es/."
        )
        self._context_prompt = (
            "Aquí tienes piezas de información (texto o imágenes) extraídas de documentos "
            "que tal vez sean relevantes para la consulta del usuario."
        )

    # ------------------------- RAG core --------------------------------------
    def _answer_question(
        self, question: str, img_paths: List[str] = None, history_msgs: List = None
    ) -> AIMessage:
        # 1) Recuperar contexto
        docs: List[Document] = self._retriever.invoke(question, img_paths)

        # context_imgs = []
        docs_contents = []
        for d in docs:
            docs_contents.append(d.page_content)
            # context_imgs.extend(extract_img_urls(d.page_content))

        context_msg = build_context_message(self._context_prompt, docs_contents)

        # 2) Construir prompt + partes de imagen
        user_msg = build_user_message(question, img_paths)
        system_msg = build_system_message(self._system_prompt)

        # 3) Construir los mensajes al modelo junto con el historial para que tenga memoria
        messages_to_model: List[BaseMessage] = [system_msg]
        if context_msg:
            messages_to_model.append(context_msg)
        if history_msgs:
            messages_to_model.extend(history_msgs)
        messages_to_model.append(user_msg)

        # 4) Llamar al LLM
        return self._llm.invoke(messages_to_model)
    
    def ask(self, query: str, imgs_path: List[str] = None) -> str:
        answer_msg = self._answer_question(query, imgs_path)
        return answer_msg.content

    # ------------------------- API pública -----------------------------------
    async def ask_stream(self, query: str, 
                         imgs_path: List[str] = None,
                         history: List[Any] = None) -> AsyncIterator[str]:
        """
        Devuelve tokens (trozos de texto) para poder ‘streamearlos’ al front.
        Guarda la memoria al finalizar.
        """
        # Construimos los mensajes
        docs = await self._retriever.ainvoke(query, imgs_path)

        context_imgs: List[str] = []
        docs_contents: List[str] = []
        for d in docs:
            docs_contents.append(d.page_content)
            # context_imgs.extend(extract_img_urls(d.page_content))

        context_msg = build_context_message(self._context_prompt, docs_contents)
        user_msg = build_user_message(query, imgs_path)
        system_msg = build_system_message(self._system_prompt)

        messages_to_model: List[BaseMessage] = [system_msg]
        if context_msg:
            messages_to_model.append(context_msg)
        if history:
            messages_to_model.extend(history)
        messages_to_model.append(user_msg)

        # LangChain-Ollama soporta .stream/.astream; usamos astream (async)
        # Cada chunk suele ser AIMessageChunk con .content
        full = []
        async for chunk in self._llm.astream(messages_to_model):
            piece = getattr(chunk, "content", "") or ""
            if piece:
                full.append(piece)
                yield piece
    
    def benchmark(self, test):
        '''
        Función para evaluar el rendimiento del asistente por medio de
        cuestionarios encapsulados en un diccionario de la forma
        {
            "questions": [],
            "options": [],
            "correct": []
        }
        '''

        text = ""
        for q, o, c in zip(test["questions"], test["options"], test["correct"]):
            query = (
                "A continuación se te proporciona una pregunta multirrespuesta:\n"
                f"{q}\n"
                "Estas son las posibles respuestas, escoge la correcta:\n"
                f"{o}\n"
            )

            text += "\n=====================================================\n"

            start = time.time()
            model_answer = self.ask(query)
            end = time.time()

            text += f"Respuesta del asistente: {model_answer}\n"
            text += f"Respuesta correcta: {c}\n"
            text += f"Latencia: {end - start:.6f}\n"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self._model_name.name}_k{self._top_k}_{timestamp}.txt"
        with open(os.path.join(self._benchmark_dir, filename), "w", encoding="utf-8") as f:
            f.write(text)
