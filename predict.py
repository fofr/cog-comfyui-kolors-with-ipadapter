# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper
from comfyui_enums import IPADAPTER_WEIGHT_TYPE, SCHEDULERS, SAMPLERS

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")
api_json_file = "workflow_api.json"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[],
        )

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    def update_workflow(self, workflow, **kwargs):
        empty_latent_image = workflow["9"]["inputs"]
        empty_latent_image["width"] = kwargs["width"]
        empty_latent_image["height"] = kwargs["height"]
        empty_latent_image["batch_size"] = kwargs["number_of_images"]

        positive_prompt = workflow["67"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]

        negative_prompt = workflow["62"]["inputs"]
        negative_prompt["text"] = f"nsfw, {kwargs['negative_prompt']}"

        sampler = workflow["79"]["inputs"]
        sampler["seed"] = kwargs["seed"]
        sampler["sampler_name"] = kwargs["sampler"]
        sampler["scheduler"] = kwargs["scheduler"]
        sampler["cfg"] = kwargs["cfg"]
        sampler["steps"] = kwargs["steps"]

        if kwargs["image_filename"]:
            load_image = workflow["95"]["inputs"]
            load_image["image"] = kwargs["image_filename"]

            ip_adapter = workflow["96"]["inputs"]
            ip_adapter["weight_type"] = kwargs["ip_adapter_weight_type"]
            ip_adapter["weight"] = kwargs["ip_adapter_weight"]
        else:
            sampler["model"] = ["59", 0]
            del workflow["93"]
            del workflow["95"]

    def predict(
        self,
        prompt: str = Input(
            default="",
        ),
        negative_prompt: str = Input(
            description="Things you do not want to see in your image",
            default="",
        ),
        image: Path = Input(
            description="Image to use as a reference for the IPAdapter",
            default=None,
        ),
        number_of_images: int = Input(
            description="Number of images to generate",
            default=1,
            ge=1,
            le=10,
        ),
        width: int = Input(
            description="Width of the image",
            default=1024,
            ge=512,
            le=2048,
        ),
        height: int = Input(
            description="Height of the image",
            default=1024,
            ge=512,
            le=2048,
        ),
        steps: int = Input(
            description="Number of inference steps",
            default=25,
            ge=1,
            le=50,
        ),
        cfg: float = Input(
            description="Guidance scale",
            default=4,
            ge=0,
            le=20,
        ),
        sampler: str = Input(
            description="Sampler",
            default="dpmpp_2m_sde_gpu",
            choices=SAMPLERS,
        ),
        scheduler: str = Input(
            description="Scheduler",
            default="karras",
            choices=SCHEDULERS,
        ),
        ip_adapter_weight_type: str = Input(
            description="Weight type for the IPAdapter",
            default="style transfer precise",
            choices=IPADAPTER_WEIGHT_TYPE,
        ),
        ip_adapter_weight: float = Input(
            description="Strength of the IPAdapter",
            default=1.0,
            ge=0,
            le=1,
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        seed = seed_helper.generate(seed)

        image_filename = None
        if image:
            image_filename = self.filename_with_extension(image, "image")
            self.handle_input_file(image_filename)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_filename=image_filename,
            number_of_images=number_of_images,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            ip_adapter_weight_type=ip_adapter_weight_type,
            ip_adapter_weight=ip_adapter_weight,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )
