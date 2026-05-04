import os, torch, shutil, asyncio
from tqdm import tqdm
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from typing import List, Dict
from PIL import Image as ImagePIL
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from huggingface_hub import snapshot_download
from pathlib import Path


def remove_dir_content(dir_path: str | Path):
    p = Path(dir_path).resolve()

    if p == Path("/") or p == Path.home():
        raise ValueError(f"Ruta peligrosa: {p}")

    if not p.is_dir():
        raise ValueError(f"No es un directorio: {p}")

    for entry in p.iterdir():
        if entry.is_file() or entry.is_symlink():
            entry.unlink(missing_ok=True)   # borra archivos/enlaces
        elif entry.is_dir():
            shutil.rmtree(entry)            # borra subcarpetas recursivamente

class MultimodalEmbedding(Embeddings):
    """Embeddings (texto + imagen) con ColQwen2.5 (un vector por texto o imagen)."""

    def __init__(self):
        self._model_name = "vidore/colqwen2.5-v0.2"
        self._device = "cpu"
        self._model = None
        self._proc = None
        self._load()

    def _load(self):
        local_path = "models/colqwen2.5-v0.2"
        if not os.path.isdir(local_path):
            snapshot_download(self._model_name, local_dir=local_path)
            self._proc = ColQwen2_5_Processor.from_pretrained(
                self._model_name, use_fast=True, torch_dtype="auto",
                low_cpu_mem_usage=True, device_map=self._device
            )
            self._model = ColQwen2_5.from_pretrained(
                self._model_name, low_cpu_mem_usage=True,
                torch_dtype="auto", device_map=self._device,
            ).eval()
        else:
            self._proc = ColQwen2_5_Processor.from_pretrained(
                local_path, use_fast=True, torch_dtype="auto",
                low_cpu_mem_usage=True, local_files_only=True, device_map=self._device
            )
            self._model = ColQwen2_5.from_pretrained(
                local_path, low_cpu_mem_usage=True, torch_dtype="auto",
                device_map=self._device, local_files_only=True
            ).eval()

    def _mean_pool(self, tensor):
        return torch.nn.functional.normalize(tensor, dim=-1).mean(dim=1)

    # ---- interfaz LangChain ----
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch = self._proc.process_queries(texts).to(self._model.device)
        with torch.no_grad():
            toks = self._model(**batch)
        return self._mean_pool(toks).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_image(self, uris: List[str]) -> List[List[float]]:
        vectors = []
        for p in uris:
            img = ImagePIL.open(p).convert("RGB")
            batch = self._proc.process_images([img]).to(self._model.device)
            with torch.no_grad():
                patches = self._model(**batch)
            vec = torch.nn.functional.normalize(patches, dim=-1).mean(dim=1)
            vectors.append(vec.squeeze(0).tolist())
        return vectors

class DataRetriever():
    """Encapsula la BD vectorial con MultiVectorRetriever (padre + hijos)."""

    def __init__(self, data_loader, persist_dir, top_k=6, assistant_mode=True):
        super().__init__()
        self._data_loader = data_loader
        self._persist_dir = persist_dir
        self._top_k = top_k
        self._embed = MultimodalEmbedding()

        self._store_dir = os.path.join(self._persist_dir, "blob_store")
        self._vs_text_dir = os.path.join(self._persist_dir, "vectorstore")

        if not os.path.exists(self._store_dir):
            raise Exception("El directorio del vectorstore no existe.")
        if not os.path.exists(self._vs_text_dir):
            raise Exception("El directorio del docstore no existe.")

        if assistant_mode:
            self._load_retrievers()
        else:
            self._create_retrievers()

    def _add_elements(self, data):
        parents: List[Document] = data["Parents"]
        child_texts: List[Document] = data["ChildTexts"]
        child_images: List[Dict[str, str]] = data["ChildImages"]

        MAX_BATCH_CHILDS = 300
        # 1) Hijos TEXT
        with tqdm(total=len(child_texts), desc="Hijos texto añadidos al vectorstore") as pbar:
            for i in range(0, len(child_texts), MAX_BATCH_CHILDS):
                batch = child_texts[i:i+MAX_BATCH_CHILDS]
                self._vs.add_documents(batch)
                pbar.update(len(batch))

        MAX_BATCH_IMGS = 100
        # 2) Hijos IMAGE -> add_images
        with tqdm(total=len(child_images), desc="Hijos imagen añadidos al vectorstore") as pbar:
            uris, metas = [], []
            for ci in child_images:
                uris.append(ci["path"])
                metas.append({"doc_id": ci["doc_id"], 
                              "type": "image", 
                              "image_path": ci["path"]})
            for i in range(0, len(uris), MAX_BATCH_IMGS):
                u_batch = uris[i:i+MAX_BATCH_IMGS]
                m_batch = metas[i:i+MAX_BATCH_IMGS]
                self._vs.add_images(u_batch, metadatas=m_batch, ids=None)  # dejar que Chroma asigne ids
                pbar.update(len(u_batch))

        # 3) Guardar SOLO los padres en el docstore (clave = doc_id)
        self._retriever.docstore.mset([(doc.metadata["doc_id"], doc) for doc in parents])

    def _create_retrievers(self):
        print("Creando el Data Retriever...\n")

        remove_dir_content(self._store_dir)
        remove_dir_content(self._vs_text_dir)

        data = self._data_loader.process_elements()

        self._vs = Chroma(
            collection_name="vectorstore",
            embedding_function=self._embed,
            persist_directory=self._vs_text_dir,
        )

        raw_store = LocalFileStore(self._store_dir)
        kv_docstore = create_kv_docstore(raw_store)

        self._retriever = MultiVectorRetriever(
            vectorstore=self._vs,
            docstore=kv_docstore,
            id_key="doc_id",
        )

        print("Añadiendo elementos...")
        self._add_elements(data)
        print("--- DATA RETRIEVER CREADO -----------------------------------")

    def _load_retrievers(self):
        print("Recuperando el Data Retriever...\n")

        if not os.listdir(self._store_dir) or not os.listdir(self._vs_text_dir):
            raise Exception("Los directorios de los stores están vacíos.")

        self._vs = Chroma(
            collection_name="vectorstore",
            embedding_function=self._embed,
            persist_directory=self._vs_text_dir,
        )

        raw_store = LocalFileStore(self._store_dir)
        kv_docstore = create_kv_docstore(raw_store)

        self._retriever = MultiVectorRetriever(
            vectorstore=self._vs,
            docstore=kv_docstore,
            id_key="doc_id",
        )

    @staticmethod
    def _numeric_score(value):
        if value is None:
            return float("-inf")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("-inf")

    @staticmethod
    def _doc_key(doc, metadata):
        doc_id = metadata.get("doc_id")
        if doc_id:
            return doc_id
        attr_id = getattr(doc, "id", None)
        if attr_id:
            return attr_id
        return f"fallback::{hash(getattr(doc, 'page_content', None))}"

    def _merge_results(self, aggregated_results):
        """
        Deduplica y ordena documentos combinando puntuaciones y procedencia.
        aggregated_results: Iterable de (source_idx, docs)
        """
        unique = {}
        for source_idx, docs in aggregated_results:
            for rank, doc in enumerate(docs):
                metadata = getattr(doc, "metadata", {}) or {}
                key = self._doc_key(doc, metadata)
                score = self._numeric_score(metadata.get("score"))
                candidate = {
                    "doc": doc,
                    "score": score,
                    "rank": rank,
                    "source_idx": source_idx,
                }

                current = unique.get(key)
                if current is None:
                    unique[key] = candidate
                    continue

                if candidate["score"] > current["score"]:
                    unique[key] = candidate
                elif candidate["score"] == current["score"]:
                    if candidate["rank"] < current["rank"]:
                        unique[key] = candidate
                    elif (
                        candidate["rank"] == current["rank"]
                        and candidate["source_idx"] < current["source_idx"]
                    ):
                        unique[key] = candidate

        ordered = sorted(
            unique.values(),
            key=lambda item: (-item["score"], item["rank"], item["source_idx"]),
        )
        return [item["doc"] for item in ordered[: self._top_k]]

    async def _ainvoke_single(self, payload, kwargs):
        if hasattr(self._retriever, "ainvoke"):
            return await self._retriever.ainvoke(payload, **kwargs)
        return await asyncio.to_thread(self._retriever.invoke, payload, **kwargs)

    def invoke(self, query, imgs=None, **kwargs):
        """
        Devuelve los documentos (padres) relevantes.
        Las imágenes se consultan como señales adicionales opcionales.
        """
        retrieval_inputs = [query]
        if imgs:
            retrieval_inputs.extend(imgs)

        aggregated = []
        for source_idx, payload in enumerate(retrieval_inputs):
            docs = self._retriever.invoke(payload, **kwargs)
            aggregated.append((source_idx, docs))
        return self._merge_results(aggregated)

    async def ainvoke(self, query, imgs=None, **kwargs):
        """
        Versión asíncrona de invoke para consultas concurrentes.
        """
        retrieval_inputs = [query]
        if imgs:
            retrieval_inputs.extend(imgs)

        tasks = [
            self._ainvoke_single(payload, kwargs)
            for payload in retrieval_inputs
        ]
        results = await asyncio.gather(*tasks)
        aggregated = list(enumerate(results))
        return self._merge_results(aggregated)

    def get_retriever(self):
        return self._retriever

    def get_top_k(self):
        return self._top_k
