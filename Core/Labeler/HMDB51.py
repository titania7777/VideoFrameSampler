import os
from glob import glob
from Core.utils import path_manager

"""
This function working for "HMDB51" dataset
Makes custom train/val/(test) split for categorical learning

The output CSV file header: ["subdirectory path of video file", "label", "category"]

[HMDB51 official annotations]
http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar
"""

def run(official_split_path:str, save_path:str, id:int=1):
    # path checking(1)
    path_manager(official_split_path, raise_error=True, path_exist=True)

    # split id => 1, 2, 3
    assert id in [1, 2, 3], f"'{id}' is not supprted split id on HMDB51 :("
    train_csv_path = os.path.join(save_path, f"train_{id}.csv")
    val_csv_path = os.path.join(save_path, f"val_{id}.csv")
    test_csv_path = os.path.join(save_path, f"test_{id}.csv")

    if path_manager(save_path, raise_error=False, create_new=True):
        print(f"{save_path} path already exists skip this step...")
        return train_csv_path, val_csv_path, test_csv_path

    # path checking(2)
    path_manager(train_csv_path, val_csv_path, test_csv_path, remove_response=True)
    
    # ready for writing
    train_csv = open(f"{train_csv_path}", "w")
    val_csv = open(f"{val_csv_path}", "w")
    test_csv = open(f"{test_csv_path}", "w")
    
    # for indexing
    categories = []
    label = 0

    for text_filename in glob(os.path.join(official_split_path, "*")):
        splited_filename = (text_filename.split("/")[-1]).split("_")
        category = "_".join(splited_filename[:-2])
        text_id= splited_filename[-1][5:-4] # get siplit id from text filename

        # get information from each same split id and different categories
        if category not in categories and int(text_id) == id:
            with open(text_filename, "r") as f:
                for line in f.read().splitlines():
                    video_filename, video_id = line.split(" ")[:-1]
                    
                    # category/video_filename
                    video_file_path = os.path.join(category, video_filename[:-4])
                    
                    # https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/split_readme.txt
                    # train
                    if int(video_id) == 1:
                        train_csv.writelines(f"{video_file_path},{label},{category}\n")
                    
                    # test
                    if int(video_id) == 2:
                        test_csv.writelines(f"{video_file_path},{label},{category}\n")
                    
                    # validation
                    if int(video_id) == 0:
                        val_csv.writelines(f"{video_file_path},{label},{category}\n")

            categories.append(category)
            label += 1
    
    # close
    train_csv.close()
    val_csv.close()
    test_csv.close()

    return train_csv_path, val_csv_path, test_csv_path