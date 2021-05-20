import os
import cv2
import ffmpeg
from Core.utils import path_manager
from glob import glob
from joblib import Parallel, delayed

def run(videos_path:str, save_path:str, frame_size:int, qscale:float, workers:int, original_size:bool):
    # path checking
    path_manager(videos_path, raise_error=True, path_exist=True)
    if path_manager(save_path, raise_error=False, path_exist=True):
        print(f"{save_path} path already exists skip this step...")
        return

    # get videos path (start point of the path is using for flexible path parsing)
    start_point_of_path = len(os.path.join(videos_path, "hello").split("/")) - 1
    videos_path = glob(os.path.join(videos_path, "**/*.*"), recursive=True)

    # run ~
    Parallel(n_jobs=workers, backend="threading")(delayed(frame_extractor)(
        [i, len(videos_path)], video_path, start_point_of_path, save_path, frame_size, qscale, original_size
    ) for i, video_path in enumerate(videos_path))

def frame_extractor(index:int, video_path:str, start_point_of_path:int, frame_path:str, frame_size:int, qscale:int, original_size:bool):
    # get a filename and make a save directory
    filename, frame_path = get_filename_savepath(start_point_of_path, video_path, frame_path)

    # get a frame information
    width_original, height_original, length = get_frame_info(video_path)

    # resizing
    if original_size:
        width_resize, height_resize = width_original, height_original
    else:
        width_resize, height_resize = frame_resizing(width_original, height_original, frame_size)

    # message
    print(f"{index[0]+1}/{index[1]} ({width_original}x{height_original}) -> ({width_resize}x{height_resize}) length: {length:<{5}} name: {filename}")

    # read and save
    (
        ffmpeg.input(video_path)
        .filter("scale", width_resize, height_resize)
        .output(os.path.join(frame_path, "%d.jpeg"), qscale=qscale)
        .global_args("-loglevel", "error", "-threads", "1", "-nostdin")
        .run()
    )

def frame_resizing(width:int, height:int, frame_size) -> list:
    if width > height:
        aspect_ratio = width / height
        if height >= frame_size:
            height = frame_size
        width = int(aspect_ratio*height)
    else:
        aspect_ratio = height / width
        if width >= frame_size:
            width = frame_size
        height = int(aspect_ratio*width)
    return [width, height]

def get_frame_info(video_path:str) -> (int, int, int):
    # read a several informations
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height, length

def get_filename_savepath(start_point_of_path:int, video_path:str, save_path:str) -> (str, str):
    # split a video path
    video_path = video_path.split("/")

    # get a filename
    filename = video_path[-1].split(".")[0]

    # make a save directory
    save_path = os.path.join(save_path, *video_path[start_point_of_path:-1], filename)
    os.makedirs(save_path)
    return filename, save_path