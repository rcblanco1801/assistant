import os, base64, uuid, gc, tiktoken, json
from unstructured.partition.pdf import partition_pdf
from PIL import ImageFile
from langchain.schema import Document
from unstructured.documents.elements import Table
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers.fix import OutputFixingParser
from tqdm import tqdm
from typing import List
from modules.image_describer import ImageDescriber


ENC = tiktoken.get_encoding("cl100k_base")
TOKENS_PER_QA = 1000
MIN_QA, MAX_QA = 5, 100

def to_b64(path):
    """Codifica una imagen en base-64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def num_tokens(text: str) -> int:
    """Cuenta tokens con tiktoken; evita rule‑of‑thumb inexactas."""
    return len(ENC.encode(text))

def qa_count_for_chunk(text: str) -> int:
    """Calcula cuántas preguntas generar según longitud del chunk."""
    n = max(MIN_QA, min(MAX_QA, num_tokens(text) // TOKENS_PER_QA))
    return n
    
class QA(BaseModel):
    question: str = Field(..., description="Pregunta en español")
    answer: str = Field(..., description="Respuesta en español")

class QAList(BaseModel):
    items: List[QA]

class DataGenerator:
    '''Clase que procesa los pdfs a través de la librería unstructured.'''
    def __init__(self, files_dir, images_dir, json_out):
        self._files_dir = files_dir
        self._images_dir = images_dir
        self._json_out = json_out
        self._img_describer = ImageDescriber(inference=False)
        self._llm = ChatOllama(
            model="gpt-oss:20b",
            reasoning=True,        # captura el razonamiento aparte (no lo imprime en la salida)
            temperature=0.2
        )
        self._base_parser = PydanticOutputParser(pydantic_object=QAList)
        self._parser = OutputFixingParser.from_llm(
            parser=self._base_parser,
            llm=self._llm,     
            max_retries=5
        )

    def _process_pdf(self, file_path):
        pdf_elements = partition_pdf(
            file_path,
            strategy="hi_res",
            chunking_strategy="basic",
            max_characters=250000,
            new_after_n_chars=200000,
            overlap = 2000,
            include_page_breaks=False,
            languages=["eng", "spa"],
            infer_table_structure=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=False,
            extract_images_in_pdf=True,
            image_output_dir_path=self._images_dir,
        )

        return pdf_elements
    
    def _build_message(self, text):
        """
        Crea un mensaje con un texto‑prompt al final pidiendo Q/A en JSON.
        """
        msg = [
            SystemMessage(content=(
                "Eres un experto pedagogo con la función de extraer pares pregunta/respuesta "
                "de documentos procedentes del contexto de la administración pública. "
                "Reasoning: high"
            )),
            HumanMessage(content=(
                f"A partir del siguiente fragmento de texto "
                f"que surge dentro del contexto de manuales y documentos sobre el uso de "
                f"herramientas electrónicas para la administración pública, "
                f"genera {qa_count_for_chunk(text)} pares pregunta/respuesta en español. "
                f"Devuélvelos como JSON cumpliendo el siguiente formato: "
                f"{self._parser.get_format_instructions()}\n\nFragmento: "
                f"{text}\n\nJSON:")
            ),
        ]

        return msg
    
    def _invoke_llm(self, text):
        with open(self._json_out, "w", encoding="utf-8") as fout:
            # Mensaje
            msg = self._build_message(text)
            # Respuesta estructurada garantizada
            qa_list: QAList = self._parser.parse(self._llm.invoke(msg).content)
            # Serializamos cada QA por línea
            for qa in qa_list.items:
                fout.write(json.dumps(qa.dict(), ensure_ascii=False) + "\n")

    def generate_qa(self):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        files_name = os.listdir(self._files_dir)
        files = [os.path.join(self._files_dir, f) for f in files_name]

        print("Procesando los archivos para el Data Generator...")

        with tqdm(total=len(files), desc="PDFs procesados") as pbar_1:
            for pdf in files:
                elements = self._process_pdf(pdf)
                with tqdm(total=len(elements), desc="Progreso del PDF") as pbar_2:
                    # Concatenamos las descripciones de las imágenes
                    # con el texto cercano para que el modelo pueda procesarlas
                    proc_elements = []
                    for i in range(len(elements)):
                        if getattr(elements[i], "text", None):
                            neighbors = elements[max(i-3, 0):min(i+3, len(elements)-1)]
                            blob = getattr(elements[i], "text")
                            for n in neighbors:
                                if getattr(n, "image_path", None):
                                    desc = self._img_describer.describe(getattr(n, "image_path"))
                                    blob = "\n".join(blob, desc)
                            proc_elements.append(blob)
                    for el in proc_elements:
                        self._invoke_llm(el)
                        pbar_2.update(1)
                pbar_1.update(1)