import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import torch
from utils.helper import set_device
from src.glm_4_9b import GLM_4_9B

# set device
device = set_device()
dtype = torch.bfloat16
# init & setup model
glm4 = GLM_4_9B(device, dtype)
glm4.setup()
# test 1
query1 = "世界上最高的山叫什么?"
print("glm-4-9b: " + glm4.infer(query1))
# test 2
query2 = "这个山在哪个国家?"
print("glm-4-9b: " + glm4.infer(query2))
# release
glm4.cleanup()
