import os
from Core.utils import path_manager

"""
This function working for "UCF101" dataset
Makes custom train/val/(test) splits for categorical learning

The output CSV file header: ["subdirectory path of video file", "label", "category"]

[UCF101 official annotations]
https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
"""

def run(official_split_path:str, save_path:str, id:int=1):
    # path checking
    path_manager(official_split_path, raise_error=True, path_exist=True)
    path_manager(save_path, create_new=True)
    
    # split id => 1, 2, 3
    assert id in [1, 2, 3], f"'{id}' is not supported split id on UCF101 :("
    train_csv_path = os.path.join(save_path, f"train_{id}.csv")
    test_csv_path = os.path.join(save_path, f"test_{id}.csv")

    # path checking
    path_manager(train_csv_path, test_csv_path, remove_response=True)

    categories = {}
    # train
    with open(f"{train_csv_path}", "w") as f1:
        # trainlist01, trainlist02, trainlist03
        with open(os.path.join(official_split_path, f"trainlist0{id}.txt"), "r") as f2:
            for line in f2.read().splitlines():
                splited_line = line.split(" ")
                category, filename = splited_line[0].split("/")
                label = int(splited_line[1]) -1

                # indexing for test
                if category not in categories:
                    categories[category] = label
                
                # save
                f1.writelines(f"{splited_line[0][:-4]},{label},{category}\n")
    
    # test
    with open(f"{test_csv_path}", "w") as f1:
        # testlist01, testlist02, testlist03
        with open(os.path.join(official_split_path, f"testlist0{id}.txt"), "r") as f2:
            for line in f2.read().splitlines():
                category, filename = line.split("/")
                label = categories[category]

                # save
                f1.writelines(f"{line[:-4]},{label},{category}\n")
    
    return train_csv_path, None, test_csv_path