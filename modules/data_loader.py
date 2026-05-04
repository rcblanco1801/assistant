from unstructured.partition.pdf import partition_pdf
from PIL import ImageFile
from langchain.schema import Document
from unstructured.documents.elements import Image, Table
from modules.image_describer import ImageDescriber
from tqdm import tqdm
import os, base64, uuid, gc, pathlib, re, json
from typing import Dict, List, Tuple, Any, Iterator
from pathlib import Path

from unstructured.staging.base import dict_to_elements
try:
    from unstructured.staging.base import elements_to_dicts 
except ImportError:
    from unstructured.staging.base import convert_to_dict as elements_to_dicts 


CHECKPOINT_PATH = Path("checkpoints")
MAX_TOKENS_PER_DOC = 750

# ------------------- Utilidades -------------------
def num_tokens(text: str) -> int:
    '''Heurística para calcular el número de tokens de un texto'''
    # Número de caracteres
    n_chars = len(text)
    # Número de palabras
    words = re.findall(r"\w+", text)
    n_words = len(words)
    # Estimaciones desde los caracteres y desde las palabras
    est_chars = n_chars / 4.0
    est_words = n_words * 1.3
    # Promedio de ambos
    estimate = (est_chars + est_words) / 2
    return int(estimate)


def safe_stem(path: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", pathlib.Path(path).stem)

def make_doc_id(file_stem: str) -> str:
    raw = f"{file_stem}-{str(uuid.uuid4().hex[:8])}"  
    return raw

def decode_and_save_b64(b64: str, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64))
    return out_path

def image_path_from_meta(meta: Dict[str, Any], fallback_path: str) -> str:
    # 1) si unstructured guardó archivo
    if "image_path" in meta and meta["image_path"]:
        return meta["image_path"]
    # 2) si vino en base64 (clave exacta depende de versión; buscamos una razonable)
    for k in meta.keys():
        if "image_base64" in k.lower():
            return decode_and_save_b64(meta[k], fallback_path)
    # 3) si no hay nada, devolvemos el fallback (no creado)
    return fallback_path

# ------------------- Funciones para guardado de elementos de Unstructured -------------------
def save_elements_as_json(element_tuples, json_path: str | Path):
    '''Guarda las tuplas de elementos que genera partition_pdf en un json.'''
    serializable = []
    for pdf_path, elements in element_tuples:
        serializable.append({
            "pdf_path": pdf_path,
            "elements": elements_to_dicts(elements),
        })
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

def load_elements_from_json(json_path: str | Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    element_tuples = []
    for rec in data:
        element_tuples.append((rec["pdf_path"], dict_to_elements(rec["elements"])))
    return element_tuples

# ------------------- Funciones para el guardado de documentos de Langchain -------------------
def append_jsonl(path: str | Path, record: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())

def doc_to_dict(doc: Document) -> Dict[str, Any]:
    """Document -> dict JSON-serializable."""
    rec = {
        "page_content": getattr(doc, "page_content", "") or "",
        "metadata": getattr(doc, "metadata", {}) or {},
    }
    doc_id = getattr(doc, "id", None)
    if doc_id is not None:
        rec["id"] = doc_id
    return rec

def dict_to_doc(rec: Dict[str, Any]) -> Document:
    """dict -> Document, tolerante con versiones sin parámetro 'id'."""
    try:
        return Document(
            page_content=rec.get("page_content", "") or "",
            metadata=rec.get("metadata", {}) or {},
            id=rec.get("id"),
        )
    except TypeError:
        # Para versiones antiguas sin 'id' en el constructor
        return Document(
            page_content=rec.get("page_content", "") or "",
            metadata=rec.get("metadata", {}) or {},
        )

def dump_docs_jsonl(path: str | Path, docs) -> None:
    for d in docs:
        append_jsonl(path, doc_to_dict(d))

def dump_imgs_jsonl(path: str | Path, rows) -> None:
    for row in rows:
        append_jsonl(path, row)

def iter_imgs_jsonl(path: str | Path) -> Iterator[Dict[str, str]]:
    """Devuelve un dict por línea (omite líneas en blanco)."""
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)  # un objeto JSON por línea (JSONL)

def iter_docs_jsonl(path: str | Path) -> Iterator[Document]:
    """Devuelve un Document por línea (sin cargar todo en memoria)."""
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            yield dict_to_doc(rec)

def read_docs_jsonl(path: str | Path) -> List[Document]:
    """Carga todos los Documents del JSONL en una lista."""
    return list(iter_docs_jsonl(path))

def read_imgs_jsonl(path: str | Path) -> List[Dict[str, str]]:
    """Carga todos los Documents del JSONL en una lista."""
    return list(iter_imgs_jsonl(path))

class DataLoader:
    """Procesa PDFs con unstructured y construye secciones (padres) + hijos (tablas/imágenes)."""
    def __init__(self, files_dir: str, images_dir: str):
        self._files_dir = files_dir
        self._images_dir = images_dir
        self._img_describer = ImageDescriber()
        os.makedirs(self._images_dir, exist_ok=True)

    def _process_pdf(self, file_path: str):
        '''Particiona pdfs.'''
        pdf_elements = partition_pdf(
            file_path,
            strategy="hi_res",
            languages=["eng", "spa"],
            infer_table_structure=False,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,     
        )
        return pdf_elements

    def _get_elements(self) -> List[Tuple[str, list]]:
        """Devuelve [(file_path, elements), ...] para mantener contexto por archivo."""
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        files_name = os.listdir(self._files_dir)
        files = [os.path.join(self._files_dir, f) for f in files_name]
        print("Procesando los archivos para el Data Loader...")

        out = []
        with tqdm(total=len(files), desc="PDFs procesados") as pbar:
            for pdf in files:
                out.append((pdf, self._process_pdf(pdf)))
                pbar.update(1)

        print("Guardandos los elementos en json...")
        save_elements_as_json(out, CHECKPOINT_PATH / Path("element_tuples.jsonl"))

        return out

    # -------------------- público -----------------------
    def process_elements(self):
        if not (CHECKPOINT_PATH / Path("element_tuples.jsonl")).is_file():
            element_tuples = self._get_elements()
        else:
            print("Cargando los elementos del json...")
            element_tuples = load_elements_from_json(CHECKPOINT_PATH / Path("element_tuples.jsonl"))

        print("Generando los elementos y documentos para el Data Retriever...")
        parents: List[Document] = []
        child_texts: List[Document] = []
        child_images: List[Dict[str, str]] = []

        parents_path = CHECKPOINT_PATH / Path("parents.jsonl")
        child_images_path = CHECKPOINT_PATH / Path("child_images.jsonl")
        child_texts_path = CHECKPOINT_PATH / Path("child_texts.jsonl")

        if parents_path.is_file():
            parents.extend(read_docs_jsonl(parents_path))
        if child_texts_path.is_file():
            print("Recuperando los documentos y elementos del checkpoint...")
            child_texts.extend(read_docs_jsonl(child_texts_path))
        if child_images_path.is_file():
            child_images.extend(read_imgs_jsonl(child_images_path))

        # element_tuples = element_tuples[
        #     (int((CHECKPOINT_PATH / Path("current_pdf.txt")).
        #         read_text(encoding="utf-8").strip()) + 1):
        # ]
        current_pdf = int((
                CHECKPOINT_PATH / Path("current_pdf.txt")
            ).read_text(encoding="utf-8").strip()
        ) + 1

        if current_pdf >= len(element_tuples):
            del self._img_describer; gc.collect()
            return {"Parents": parents, "ChildTexts": child_texts, "ChildImages": child_images}

        pbar_pdfs = tqdm(total=len(element_tuples), desc="PDFs procesados")
        pbar_pdfs.update(current_pdf); pbar_pdfs.refresh()
        for file_idx, (file_path, elements) in enumerate(element_tuples[current_pdf:]):
            file_stem = safe_stem(file_path)
            token_count = 0
            doc_id = make_doc_id(file_stem)
            doc_text = ""

            pdf_doc_parents: List[Document] = []
            pdf_img_dicts: List[Dict] = []
            pdf_doc_texts: List[Document] = []
            pdf_doc_imgs: List[Document] = []

            pbar_elements = tqdm(total=len(elements), 
                                 desc="Elementos del pdf actual procesados")
            for idx, el in enumerate(elements):
                if token_count >= MAX_TOKENS_PER_DOC:
                    doc = Document(
                        id=doc_id,
                        page_content=doc_text,
                        metadata={
                            "doc_id": doc_id,
                            "type": "section",
                            "source_file": file_stem,
                        },
                    )
                    parents.append(doc)
                    pdf_doc_parents.append(doc)
                    doc_id = make_doc_id(file_stem)
                    token_count = 0
                    doc_text = ""

                element_id = "element_" + make_doc_id(file_stem)
                meta = el.metadata.to_dict()

                if isinstance(el, Table) or isinstance(el, Image):
                    fallback_path = os.path.join(self._images_dir, f"img_{element_id}.png")
                    img_path = image_path_from_meta(meta, fallback_path)

                    caption = self._img_describer.describe(img_path)
                    img_tag = "<img_url>" + img_path + "</img_url>" + "\n\n"
                    img_section = img_tag + "\n<image_description>" + caption + "</image_description>\n\n"
                    doc_text += img_section
                    token_count += num_tokens(img_section)

                    if caption.strip():
                        doc = Document(
                            id=element_id,
                            page_content=caption,
                            metadata={
                                "doc_id": doc_id,
                                "type": "image_caption",
                                "source_file": file_stem,
                                "image_path": img_path,
                            },
                        )
                        child_texts.append(doc)
                        pdf_doc_imgs.append(doc)

                    img_dict = {"path": img_path, "doc_id": doc_id}
                    child_images.append(img_dict)
                    pdf_img_dicts.append(img_dict)
                else:
                    text = (getattr(el, "text", None) or "").strip()
                    doc_text += text + "\n\n"
                    token_count += num_tokens(text)

                    doc = Document(
                        id=element_id,
                        page_content=text,
                        metadata={
                            "doc_id": doc_id,
                            "type": "text",
                            "source_file": file_stem,
                        },
                    )

                    child_texts.append(doc)
                    pdf_doc_texts.append(doc)

                if idx >= len(elements) - 1:
                    doc = Document(
                        id=doc_id,
                        page_content=doc_text,
                        metadata={
                            "doc_id": doc_id,
                            "type": "section",
                            "source_file": file_stem,
                        },
                    )
                    parents.append(doc)
                    pdf_doc_parents.append(doc)

                pbar_elements.update(1); pbar_elements.refresh()
            pbar_elements.close()

            print("Generando checkpoint del archivo pdf...")
            dump_docs_jsonl(CHECKPOINT_PATH / Path("parents.jsonl"), pdf_doc_parents)
            dump_docs_jsonl(CHECKPOINT_PATH / Path("child_texts.jsonl"), pdf_doc_texts)
            dump_docs_jsonl(CHECKPOINT_PATH / Path("child_texts.jsonl"), pdf_doc_imgs)
            dump_imgs_jsonl(CHECKPOINT_PATH / Path("child_images.jsonl"), pdf_img_dicts)
            (CHECKPOINT_PATH / Path("current_pdf.txt")).write_text(f"{file_idx+current_pdf}\n", encoding="utf-8")

            pbar_pdfs.update(1); pbar_pdfs.refresh()
        pbar_pdfs.close()

        del self._img_describer; gc.collect()
        return {"Parents": parents, "ChildTexts": child_texts, "ChildImages": child_images}
