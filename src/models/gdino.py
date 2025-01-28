"""
GroundingDino Model

Notes:
Modified from https://github.com/luca-medeiros/lang-segment-anything.
"""

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from models.utils import get_device_type

device_type = get_device_type()
DEVICE = torch.device(device_type)

if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class GDINO:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base"):
        self.model_id = model_id
        self.build_model()

    def build_model(self, ckpt_path: str | None = None):
        """Build the model from the checkpoint."""
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(
            DEVICE
        )


    def predict(
        self,
        pil_images: list[Image.Image],
        text_prompt: list[str] = ["watermark."],
        box_threshold: float = 0.2,
        text_threshold: float = 0.25,
    ) -> list[dict]:
        """Predicte bounding box for a 'text prompt' of an image 'pil_images' 
        
        Args:
            pil_images: the images to find bounding boxes for.
            text_prompt: indicates what to look for
            box_threshold: Score threshold to keep object detection predictions.
            text_threshold: Score threshold to keep text detection predictions.
            
        Returns:
            A bounding box of the watermark.
        
        """
        for i, prompt in enumerate(text_prompt):
            if prompt[-1] != ".":
                text_prompt[i] += "."
        inputs = self.processor(images=pil_images, text=text_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[k.size[::-1] for k in pil_images],
        )
        # can switch back if more boxes are needed. 
        bboxes = []
        for i in range(len(results)):
            bboxes.append([int(b) for b in results[i]['boxes'][0]])
        return bboxes