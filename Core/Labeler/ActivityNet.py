import os
import json
from Core.utils import path_manager

"""
This function working for "ActivityNet" dataset
Makes custom train/val/(test) splits for categorical learning

The output CSV file header: ["subdirectory path of video file", "label", "category"]

[ActivityNet official annotations]
* test.csv in activitynet is a validation label
http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/files/activity_net.v1-2.min.json
http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json
"""

def run(official_split_path:str, save_path:str, id:int=3):
    # path checking(1)
    path_manager(official_split_path, raise_error=True, path_exist=True)

    # version id => 2(1.2), 3(1.3)
    assert id in [2, 3], f"'{id}' is not supported version id on ActivityNet :("
    json_path = os.path.join(official_split_path, f"activity_net.v1-{version}.min.json")
    train_csv_path = os.path.join(save_path, f"train{version}.csv")
    val_csv_path = os.path.join(save_path, f"val{version}.csv")

    if path_manager(save_path, raise_error=False, path_exist=True):
        print(f"{save_path} path already exists skip this step...")
        return train_csv_path, val_csv_path, None
    else:
        path_manager(save_path, create_new=True)
        
    # path checking(2)
    path_manager(train_csv_path, val_csv_path, remove_response=True)

    # load and read json
    # keylist => [database, taxonomy, version]
    # see more => http://activity-net.org/download.html
    with open(json_path, "r") as f:
        database = json.load(f)["database"]
    
    trains = []
    vals = []
    categories = []
    for vid in database:
        subset = database[vid]["subset"]
        if subset == "testing":
            continue
        
        category = database[vid]["annotations"][0]["label"]

        if not category in categories:
            categories.append(category)
        
        label = f"v_{vid},{categories.index(category)},{category}"

        # train
        if subset == "training":
            trains.append(label)
        
        # validation
        if subset == "validation":
            vals.append(label)
        
    with open(train_csv_path, "w") as f:
        f.writelines("\n".join(trains))
    
    with open(val_csv_path, "w") as f:
        f.writelines("\n".join(vals))
    
    return train_csv_path, val_csv_path, None