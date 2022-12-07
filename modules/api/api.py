from modules.api.models import *
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img, process_images
from modules.sd_samplers import all_samplers
import modules.shared as shared
import modules.sd_models as sd_models
from modules.sd_models import create_checkpoint_info, save_checkpoint_file_from_url
import uvicorn
from fastapi import Body, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, Json
import json
import io
import base64
from PIL import Image


def sampler_to_index(name): return next(filter(
    lambda row: name.lower() == row[1].name.lower(), enumerate(all_samplers)), None)


def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(encoding)))


class TextToImageResponse(BaseModel):
    images: list[str] = Field(
        default=None,
        title="Image",
        description="The generated image in base64 format."
    )
    parameters: Json
    info: Json


class ImageToImageResponse(BaseModel):
    images: list[str] = Field(
        default=None,
        title="Image",
        description="The generated image in base64 format."
    )
    parameters: Json
    info: Json


class SaveCheckpointRequest(BaseModel):
    url: str = Field(default=None, title="Checkpoint",
                     description="The checkpoint to save.")


class Api:
    def __init__(self, app, queue_lock):
        self.router = APIRouter()
        self.app = app
        self.queue_lock = queue_lock

        self.app.add_api_route("/sdapi/v1/txt2img", self.text2imgapi, methods=["POST"])
        self.app.add_api_route("/sdapi/v1/img2img", self.img2imgapi, methods=["POST"])
        self.app.add_api_route("/sdapi/v1/save_checkpoint", self.save_checkpoint, methods=["POST"])

    def text2imgapi(self, txt2imgreq: StableDiffusionTxt2ImgProcessingAPI):
        sampler_index = sampler_to_index(txt2imgreq.sampler_index)[0]
        checkpoint_filename = txt2imgreq.model_checkpoint
        checkpoint_info = None

        if sampler_index is None:
            raise HTTPException(status_code=404, detail="Sampler not found")

        if checkpoint_filename is not None:
            if checkpoint_filename.startswith("http"):
                save_checkpoint_file_from_url(
                    url=checkpoint_filename
                )
            if checkpoint_info is None:
                checkpoint_info = create_checkpoint_info(
                    url=checkpoint_filename
                )
            print("loaded new model", checkpoint_info)
            sd_models.load_model(checkpoint_info=checkpoint_info)

        populate = txt2imgreq.copy(update={  # Override __init__ params
            "sd_model": shared.sd_model,
            "sampler_index": sampler_index,
            "do_not_save_samples": True,
            "do_not_save_grid": True
        })

        p = StableDiffusionProcessingTxt2Img(**vars(populate))
        with self.queue_lock:
            processed = process_images(p)

        b64images = []
        for i in processed.images:
            buffer = io.BytesIO()
            i.save(buffer, format="png")
            b64images.append(base64.b64encode(buffer.getvalue()))

        return TextToImageResponse(images=b64images, parameters=json.dumps(vars(txt2imgreq)),
                                   info=json.dumps(processed.info))

    def img2imgapi(self, img2imgreq: StableDiffusionImg2ImgProcessingAPI):
        print(img2imgreq)

        sampler_index = sampler_to_index(img2imgreq.sampler_index)[0]
        init_images = img2imgreq.init_images
        if init_images is None:
            raise HTTPException(status_code=404, detail="Init image not found")

        checkpoint_filename = img2imgreq.model_checkpoint
        checkpoint_info = None

        if sampler_index is None:
            raise HTTPException(status_code=404, detail="Sampler not found")

        if checkpoint_filename is not None:
            if checkpoint_filename.startswith("http"):
                save_checkpoint_file_from_url(
                    url=checkpoint_filename
                )
            if checkpoint_info is None:
                checkpoint_info = create_checkpoint_info(
                    url=checkpoint_filename
                )
            print("loaded new model", checkpoint_info)
            sd_models.load_model(checkpoint_info=checkpoint_info)

        mask = img2imgreq.mask
        if mask:
            mask = decode_base64_to_image(mask)

        populate = img2imgreq.copy(update={  # Override __init__ params
            "sd_model": shared.sd_model,
            "sampler_index": sampler_index,
            "do_not_save_samples": True,
            "do_not_save_grid": True,
            "mask": mask
        })
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on

        args = vars(populate)
        args.pop('include_init_images', None)  # this is meant to be done by "exclude": True in model, but it's for a reason that I cannot determine.
        p = StableDiffusionProcessingImg2Img(**args)

        print("p arguments:")
        for k, v in vars(p).items():
            print(k, v)

        imgs = []
        for img in init_images:
            img = decode_base64_to_image(img)
            imgs = [img] * p.batch_size

        p.init_images = imgs

        with self.queue_lock:
            processed = process_images(p)

        b64images = []
        for i in processed.images:
            buffer = io.BytesIO()
            i.save(buffer, format="png")
            b64images.append(base64.b64encode(buffer.getvalue()))

        return ImageToImageResponse(images=b64images, parameters=vars(img2imgreq), info=json.dumps(processed.info))

    def save_checkpoint(self, req: SaveCheckpointRequest):
        save_checkpoint_file_from_url(url=req.url)
        return JSONResponse(status_code=200, content={"message": "OK"})

    def img2imgapi(self):
        raise NotImplementedError

    def extrasapi(self):
        raise NotImplementedError

    def pnginfoapi(self):
        raise NotImplementedError

    def launch(self, server_name, port):
        self.app.include_router(self.router)
        uvicorn.run(self.app, host=server_name, port=port)
