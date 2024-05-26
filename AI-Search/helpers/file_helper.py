from pathlib import Path
from typing import List
import os
import shutil

def save_list_to_file(lines: List[str], file_dir: str,  file_name:str):
    if not os.path.exists(file_dir):
        Path(file_dir).mkdir(parents=True, exist_ok=True)

    full_file_name = os.path.join(file_dir, file_name)
    with open(full_file_name, 'w') as f:
        for line in lines:
            f.write(f"{line}\n")

def load_list_from_file(file_dir: str,  file_name:str) -> List[str]:
    full_file_name = os.path.join(file_dir, file_name)
    with open(full_file_name) as file:
        lines = [line.rstrip() for line in file]
    return lines

def remove_directory_tree(directory: str):
    if os.path.exists(directory):
        shutil.rmtree(directory)