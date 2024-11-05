import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import stat
import subprocess
import torch
import platform
from PIL import Image

def set_device():
    print('Pytorch version', torch.__version__)
    if torch.backends.mps.is_available():
        print("set device to 'mps'")
        device = torch.device('mps')
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["ACCELERATE_USE_MPS_DEVICE"] = "True"
    elif torch.cuda.is_available():
        print("set device to 'cuda'")
        device = torch.device('cuda', 0)
    else:
        print("set device to 'cpu'")
        device = torch.device('cpu')
    return device

def show_img(file_path : str, title : str) -> None:
    if platform.system() == "Darwin":
        subprocess.run(["osascript", "-e", 'tell application "Preview" to quit'], check=False) # close last preview window
        subprocess.run(["sudo", "open", file_path]) # show current preview window
    else:
        img = Image.open(file_path)
        img.show(title)

def calc_time_consumption(start_time, end_time) -> None:
    if end_time == 0 and start_time == 0:
        print("Warning: both 'end time' and 'start time' are 0.0. no time calculation can be performed.")
        return
    elapsed_time = (end_time - start_time) / 60.0
    print(f"timer: took {elapsed_time:.2f} minutes totally")

def remove_files_except_with_suffix(folder_path : str, suffix : str) -> None:
    for file in os.listdir(folder_path):
        if not file.endswith(suffix):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

def remove_all_files(folder_path : str) -> None:
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def check_and_init_folder(folder_path : str) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        if platform.system() == "Darwin":
            os.chmod(folder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO) # 777 permission
    else:
        remove_all_files(folder_path)

def get_new_folder_name_with_index(folder_name, index):
    return f"{folder_name}/img_{index}"

def get_new_object_name_with_index(path : str, index : int) -> str:
    return f"{ path.split('.')[0] }_{ index }.{ path.split('.')[1] }"

def find_single_file_with_suffix(folder_path : str, suffix : str) -> str:
    for file in os.listdir(folder_path):
        if file.endswith(suffix):
            return os.path.join(folder_path, file)
    return None
