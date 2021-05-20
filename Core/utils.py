import os
import csv
import sys
import shutil
import torch
from torch.backends import cudnn

def path_manager(*paths, raise_error:bool = False, path_exist:bool = False, create_new:bool = False, remove_enforcement:bool = False, remove_response:bool = False) -> bool:
    for path in paths:
        if path == None:
            continue

        exist = os.path.exists(path)
        # path check
        if path_exist:
            if raise_error:
                assert exist, f"{path} is not exist :("
            else:
                if not exist:
                    return False
        
        # remove => create (possible)
        # path remove(enforcement)
        if remove_enforcement and exist:
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
        # path remove(response)
        elif remove_response and exist:
            while True:
                print(f"'{path}' is already exist, do you want to continue after remove that ? [y/n]")
                response = input()

                # yes
                if response == "y" or response == "Y" or response == "yes":
                    if os.path.isfile(path):
                        os.remove(path)
                    else:
                        shutil.rmtree(path)
                        os.makedirs(path)
                    break
            
                # no
                if response == "n" or response == "N" or response == "no":
                    print("this script was terminated by a user :/")
                    sys.exit()
        
        # path create
        if create_new and not exist:
            os.makedirs(path)

    return True

def read_csv(csv_path: str) -> (list, dict):
    path_and_labels = [] # [[sub directory file path, label]]
    categories = {} # {label: category}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for rows in reader:
            sub_file_path = rows[0]; label = int(rows[1]); category = rows[2]
            path_and_labels.append([sub_file_path, label])
            if label not in categories:
                categories[label] = category

    return path_and_labels, categories

def get_device(only_cpu:bool = False, gpu_number:int = 0, cudnn_benchmark:bool = True) -> torch.device:
    if torch.cuda.is_available() and not only_cpu:
        if cudnn_benchmark:
            cudnn.benchmark = True
        return torch.device(f"cuda:{gpu_number}")
    else:
        return torch.device("cpu")