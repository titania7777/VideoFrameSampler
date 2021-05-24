import os
import argparse
from Core import Labeler, FrameExtractor, FrameSampler
from Core.utils import path_manager

def main(official_split_path, csv_path, videos_path, frames_path, json_path, args):
    # Labeling
    if args.dataset_name == "UCF101":
        train_csv_path, val_csv_path, test_csv_path = Labeler.UCF101.run(
            official_split_path=official_split_path,
            save_path=csv_path,
            id=args.split_id
        )
    elif args.dataset_name == "HMDB51":
        train_csv_path, val_csv_path, test_csv_path = Labeler.HMDB51.run(
            official_split_path=official_split_path,
            save_path=csv_path,
            id=args.split_id
        )
    elif args.dataset_name == "ActivityNet":
        train_csv_path, val_csv_path, test_csv_path = Labeler.ActivityNet.run(
            official_split_path=official_split_path,
            save_path=csv_path,
            id=args.split_id
        )
    else:
        print(f"'{args.dataset_name}' is not supported :(")
        return

    # Frame Extraction
    FrameExtractor.run(
        videos_path=videos_path,
        save_path=frames_path,
        frame_size=args.frame_size_extractor,
        qscale=args.qscale,
        workers=args.workers,
        original_size=args.original_size
    )
    
    # Frame Sampling
    if path_manager(json_path, raise_error=False, path_exist=True):
        print(f"{json_path} path already exists skip this step...")
        return
    else:
        path_manager(json_path, create_new=True)
    for csv_path in [train_csv_path, val_csv_path, test_csv_path]:
        if csv_path:
            FrameSampler.run(
                frames_path=frames_path,
                csv_path=csv_path,
                save_path=json_path,
                frame_batch_size=args.frame_batch_size,
                frame_size=args.frame_size_sampler,
                only_cpu=args.only_cpu,
                gpu_number=args.gpu_number
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="./Data/")
    parser.add_argument("--dataset-name", type=str, default="UCF101")
    parser.add_argument("--split-id", type=int, default=1)
    # for setting the frame extractor parameters
    parser.add_argument("--frame-size-extractor", type=int, default=240)
    parser.add_argument("--qscale", type=int, default=7)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--original-size", action="store_true")
    # for setting the frame sampler parameters
    parser.add_argument("--frame-batch-size", type=int, default=500)
    parser.add_argument("--frame-size-sampler", type=int, default=112)
    parser.add_argument("--gpu-number", type=int, default=0)
    parser.add_argument("--only-cpu", action="store_true")
    args = parser.parse_args()

    # Path organize
    official_split_path = os.path.join(args.data_path, f"{args.dataset_name}/official_split/")
    csv_path = os.path.join(args.data_path, f"{args.dataset_name}/custom_split/")
    videos_path = os.path.join(args.data_path, f"{args.dataset_name}/videos/")
    frames_path = os.path.join(args.data_path, f"{args.dataset_name}/frames/")
    json_path = os.path.join(args.data_path, f"{args.dataset_name}/sampled_split/")

    main(official_split_path, csv_path, videos_path, frames_path, json_path, args)