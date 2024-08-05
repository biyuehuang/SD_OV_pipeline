from optimum.intel import OVStableDiffusionPipeline
import time
from pathlib import Path
import numpy as np
import os
from PIL import Image
from diffusers import EulerDiscreteScheduler
import requests
import torch

if 0:
    DEVICE_NAME = "GPU.1"
    scheduler = EulerDiscreteScheduler.from_pretrained("stable-diffusion-2-1-base", subfolder="scheduler")
    pipe = OVStableDiffusionPipeline.from_pretrained("stable-diffusion-2-1-base", scheduler=scheduler, device=DEVICE_NAME, export=True)
    pipe.save_pretrained("ov_models_dynamic")

NEED_STATIC = True
#STATIC_SHAPE = [512,512]
STATIC_SHAPE = [1024,1024]
DEVICE_NAME = "GPU.1"
if NEED_STATIC:
    print("Using static models")
    scheduler = EulerDiscreteScheduler.from_config("ov_models_dynamic/scheduler/scheduler_config.json")
    ov_config ={"CACHE_DIR": "",'INFERENCE_PRECISION_HINT': 'f16'}
    if not os.path.exists("ov_models_static"):
        if os.path.exists("ov_models_dynamic"):
            print("load dynamic models from local ov files and reshape to static")
            ov_pipe = OVStableDiffusionPipeline.from_pretrained(Path("ov_models_dynamic"), scheduler=scheduler, device=DEVICE_NAME, compile=True, ov_config=ov_config, height=STATIC_SHAPE[0], width=STATIC_SHAPE[1])
            ov_pipe.reshape(batch_size=1,height=STATIC_SHAPE[0], width=STATIC_SHAPE[1], num_images_per_prompt=1)
            ov_pipe.save_pretrained(save_directory="./ov_models_static")
            print("Static model is saved in ./ov_models_static")  
            exit()
        else:
            raise ValueError("No ov_models_dynamic exists, please trt ov_model_export.py first")
    else:
        print("load static models from local ov files")
        ov_pipe = OVStableDiffusionPipeline.from_pretrained(Path("ov_models_static"), scheduler=scheduler, device=DEVICE_NAME, compile=True, ov_config=ov_config, height=STATIC_SHAPE[0], width=STATIC_SHAPE[1])
else:
    scheduler = EulerDiscreteScheduler.from_config("ov_models_dynamic/scheduler/scheduler_config.json")
    ov_config ={"CACHE_DIR": "", 'INFERENCE_PRECISION_HINT': 'f16'}
    print("load dynamic models from local ov files")
    ov_pipe = OVStableDiffusionPipeline.from_pretrained(Path("ov_models_dynamic"), scheduler=scheduler, device=DEVICE_NAME, compile=True, ov_config=ov_config)

seed = 42
torch.manual_seed(seed)           
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

#set prompt, negative_prompt, num_inference_steps to get result
prompt = "Dancing Darth Vader, best quality, extremely detailed"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

#prompt = ["a photo of an astronaut riding a horse on mars","a photo of an astronaut riding a horse on mars","a photo of an astronaut riding a horse on mars","a photo of an astronaut riding a horse on mars"]
#negative_prompt = ["monochrome, lowres, bad anatomy, worst quality, low quality","monochrome, lowres, bad anatomy, worst quality, low quality","monochrome, lowres, bad anatomy, worst quality, low quality","monochrome, lowres, bad anatomy, worst quality, low quality"]

st = time.time()
for id in range(9):
    st_id = time.time()
    result = ov_pipe(prompt=prompt, num_inference_steps=20, negative_prompt=negative_prompt, height=STATIC_SHAPE[0], width=STATIC_SHAPE[1],num_images_per_prompt=1).images
   # result[0].resize((1920,1080))
    print("id, current time:",id,time.time() - st_id)
    result[0].save(f"result_1024_{id}.png")
print("total time = ",time.time()-st)
print("end")
