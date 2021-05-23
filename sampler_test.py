import os
import argparse
from torch.utils.data import DataLoader
from Core.VideoDataset import VideoDataset


def main(frames_path:str, sampled_train_split_path:str, sampled_val_split_path:str, sampled_test_split_path:str, args):
    # Training
    if os.path.exists(sampled_train_split_path):
        # Dataset
        train_dataset = VideoDataset(
            frames_path=frames_path,
            sampled_split_path=sampled_train_split_path,
            frame_size=args.frame_size,
            sequence_length=args.sequence_length,
            random_pad_sample=args.random_pad_sample
        )

        # Loader
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=0 if os.name == 'nt' else 4,
            pin_memory=True,
            shuffle=True,
        )

        print(f"Number of Training data: {len(train_dataset)}, Batch Size: {args.batch_size}")

        for i, (datas, labels) in enumerate(train_loader):
            print(f"{i}/{len(train_loader)} {datas.size()}")
    
    # Validation
    if os.path.exists(sampled_val_split_path):
        # Dataset
        val_dataset = VideoDataset(
            frames_path=frames_path,
            sampled_split_path=sampled_val_split_path,
            frame_size=args.frame_size,
            sequence_length=args.sequence_length,
            random_pad_sample=args.random_pad_sample
        )

        # Loader
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            num_workers=0 if os.name == 'nt' else 4,
            pin_memory=True,
            shuffle=False,
        )

        print(f"Number of Validation data: {len(val_dataset)}, Batch Size: {args.batch_size}")

        for i, (datas, labels) in enumerate(val_loader):
            print(f"{i}/{len(val_loader)} {datas.size()}")
    
    # Testing
    if os.path.exists(sampled_test_split_path):
        # Dataset
        test_dataset = VideoDataset(
            frames_path=frames_path,
            sampled_split_path=sampled_test_split_path,
            frame_size=args.frame_size,
            sequence_length=args.sequence_length,
            random_pad_sample=args.random_pad_sample
        )

        # Loader
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=0 if os.name == 'nt' else 4,
            pin_memory=True,
            shuffle=False,
        )

        print(f"Number of Testing data: {len(test_dataset)}, Batch Size: {args.batch_size}")

        for i, (datas, labels) in enumerate(test_loader):
            print(f"{i}/{len(test_loader)} {datas.size()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="./Data/")
    parser.add_argument("--dataset-name", type=str, default="UCF101")
    parser.add_argument("--split-id", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--frame-size", type=int, default=112)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--random-pad-sample", action="store_true")
    args = parser.parse_args()

    # Dataset check
    assert args.dataset_name in ["UCF101", "HMDB51", "ActivityNet"], f"{args.dataset_name} is not supported dataset :("

    # Path organize
    frames_path = os.path.join(args.data_path, f"{args.dataset_name}/frames/")
    sampled_train_split_path = os.path.join(args.data_path, f"{args.dataset_name}/sampled_split/train_{args.split_id}.json")
    sampled_val_split_path = os.path.join(args.data_path, f"{args.dataset_name}/sampled_split/val_{args.split_id}.json")
    sampled_test_split_path = os.path.join(args.data_path, f"{args.dataset_name}/sampled_split/test_{args.split_id}.json")

    main(frames_path, sampled_train_split_path, sampled_val_split_path, sampled_test_split_path, args)