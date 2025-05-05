from system_prompt import TABLES_AND_CHARTS_CONVERT_SYSTEM
import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from typing import Optional

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

Qwen_VL_PATH = "/root/autodl-tmp/Qwen2.5-VL-7B-Instruct"

class QwenConvertorFactory:
    _instance = None

    @staticmethod
    def get_instance(device="cuda:0" if torch.cuda.is_available() else "cpu"):
        """Static method to get the singleton instance of QwenConvertor."""
        if QwenConvertorFactory._instance is None:
            QwenConvertorFactory._instance = QwenConvertor()
        return QwenConvertorFactory._instance


class QwenConvertor:
    def __init__(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            Qwen_VL_PATH, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(Qwen_VL_PATH)

    def convert(
        self,
        img_path: str,
        system_prompt: str = TABLES_AND_CHARTS_CONVERT_SYSTEM,
    ):
        assert (
            img_path.endswith(".png")
            or img_path.endswith(".jpg")
            or img_path.endswith(".jpeg")
        ), "Only support png, jpg, jpeg format"
        assert os.path.isfile(img_path), f"{img_path} does not exist"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": system_prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]

    def clear_GPU_mem(self):
        del self.model
        del self.processor
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()


if __name__ == '__main__':
    con = QwenConvertor()
    res = con.convert(img_path="/root/autodl-tmp/demo_img.png")
    print(type(res))
    print(res)
