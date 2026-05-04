import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from huggingface_hub import snapshot_download


class ImageDescriber:
    def __init__(self):
        local_path = "models/Qwen2.5-VL-3B-Instruct"
        hf_path = "Qwen/Qwen2.5-VL-3B-Instruct"

        if not os.path.isdir(local_path):
            snapshot_download(hf_path, local_dir=local_path)
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                hf_path,
                torch_dtype="auto",
                device_map="auto",
            ).eval()
            self._proc = AutoProcessor.from_pretrained(hf_path)
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                local_path,
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True
            ).eval()
            self._proc = AutoProcessor.from_pretrained(
                local_path, local_files_only=True
            )
    
    def describe(self, image_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "A continuación se te adjunta una imagen que muy probablemente "
                        "provenga de una interfaz de usuario. Descríbela detalladamente "
                        "en español, "
                        "extrayendo todo el texto e infiriendo toda la estructura que "
                        "puedas por medio de, por ejemplo, código HTML. Si no es una "
                        "imagen de interfaz, deja la descripción vacía."
                    )},
                    {
                        "type": "image",
                        "image": image_path,
                    },
                ],
            }
        ]

        # Preparation for inference
        text = self._proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._proc(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cpu")

        # Inference: Generation of the output
        generated_ids = self._model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._proc.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]