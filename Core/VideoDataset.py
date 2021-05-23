import os
import json
import numpy as np
import torch
from PIL import Image # Try using the pillow-simd !!
from torch.utils.data import Dataset
from glob import glob
import torchvision.transforms as transforms

"""
Returns JSON file structure as follows
{
    sub_file_path: {
        label,
        category,
        index: []
    }
}
sub_file_path => subdirectory of the frames file path
label => label(index number)
category => class name corresponds to the label
index => the sampled index number of the video frames
"""
# VideoDataset
class VideoDataset(Dataset):
    def __init__(self, frames_path:str, sampled_split_path:str, frame_size:int = 112, sequence_length:int = 16, random_pad_sample:bool = False):

        self.frames_path = frames_path
        self.sequence_length = sequence_length

        # arguments for self._add_pads
        self.random_pad_sample = random_pad_sample
        
        # read a sampled split json file
        with open(sampled_split_path, "r") as f:
            self.dataset = json.load(f)
        self.sub_file_path_list = list(self.dataset)

        # transformer
        self.transform = transforms.Compose([
            transforms.Resize((frame_size, frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.5, 0.5, 0.5],
                std = [0.5, 0.5, 0.5]
            ),
        ])
    
    def __len__(self) -> int:
        return len(self.sub_file_path_list)

    def _add_pads(self, length_of_frames: int, sequence_length: int, random_pad_sample: bool) -> np.ndarray:
        # length -> array
        sequence = np.arange(length_of_frames)
        require_frames = sequence_length - length_of_frames

        if random_pad_sample:
            # random samples of frames
            add_sequence = np.random.choice(sequence, require_frames)
        else:
            # repeated a first frame
            add_sequence = np.repeat(sequence[0], require_frames)

        # sorting of the array
        sequence = sorted(np.append(sequence, add_sequence, axis=0))

        return sequence

    def __getitem__(self, index):
        sub_file_path = self.sub_file_path_list[index]
        label = self.dataset[sub_file_path]["label"]
        
        # hmdb51 has some weird filenames that can't catch when using glob
        replaced_sub_file_path = sub_file_path.replace("]", "?")

        # get frames path
        sampled_indices = sorted(self.dataset[sub_file_path]["index"][:self.sequence_length])
        images_path = np.array(sorted(glob(os.path.join(self.frames_path, replaced_sub_file_path, "*")), key=lambda file: int(file.split("/")[-1].split(".")[0])))
        images_path = images_path[sampled_indices]

        # get index of samples
        length_of_frames = len(images_path)
        assert length_of_frames != 0, f"'{sub_file_path}' is not exists or empty."

        if length_of_frames < self.sequence_length:
            indices = self._add_pads(length_of_frames, self.sequence_length, self.random_pad_sample)
            images_path = images_path[indices]

        # load frames
        data = torch.stack([self.transform(Image.open(image_path)) for image_path in images_path], dim=1)

        return data, label