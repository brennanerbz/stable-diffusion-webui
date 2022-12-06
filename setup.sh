#!/bin/bash

conda remove -n web-ui --all
conda create -n web-ui python=3.10
conda activate web-ui

pip install cython
pip install numpy
pip install "git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI"
pip install cmake
pip install pycocotools
pip install dlib
pip install psutil
pip install setuptools
pip install wheel
pip install font-roboto
pip install -r requirements.txt

#git clone https://github.com/CompVis/stable-diffusion.git repositories/stable-diffusion
#git clone https://github.com/CompVis/taming-transformers.git repositories/taming-transformers
#git clone https://github.com/sczhou/CodeFormer.git repositories/CodeFormer
#git clone https://github.com/salesforce/BLIP.git repositories/BLIP
#git clone https://github.com/Birch-san/k-diffusion repositories/k-diffusion
#
#pip install git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1
#pip install git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379
#pip install torch==1.12.1 torchvision==0.13.1

#echo "--- a/functional.py	2022-10-14 05:28:39.000000000 -0400
#+++ b/functional.py	2022-10-14 05:39:25.000000000 -0400
#@@ -2500,7 +2500,7 @@
#         return handle_torch_function(
#             layer_norm, (input, weight, bias), input, normalized_shape, weight=weight, bias=bias, eps=eps
#         )
#-    return torch.layer_norm(input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled)
#+    return torch.layer_norm(input.contiguous(), normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled)
#
#
# def group_norm(
#" | patch -p1 -d "$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))")"/nn

pip install gdown
pip install fastapi

conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1

gunicorn --bind 0.0.0.0:5001 --workers 1 --worker-class uvicorn.workers.UvicornWorker  --threads 8 webui:app --timeout 540