import os
import argparse
from Core import Labeler, FrameExtractor, FrameSampler

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
    parser.add_argument("--frame-size-sampler", type=int, default=112)
    parser.add_argument("--gpu-number", type=int, default=0)
    parser.add_argument("--only-cpu", action="store_true")
    args = parser.parse_args()

    # dataset name check
    assert args.dataset_name in ["UCF101", "HMDB51", "ActivityNet"], f"'{args.dataset_name}' is not supported :("
    
    # UCF101
    train_csv_path, val_csv_path, test_csv_path = Labeler.UCF101.run(
        official_split_path=os.path.join(args.data_path, f"{args.dataset_name}/official_split/"),
        save_path=os.path.join(args.data_path, f"{args.dataset_name}/custom_split/"),
        id=args.split_id
    )
    FrameExtractor.run(
        videos_path=os.path.join(args.data_path, f"{args.dataset_name}/videos/"),
        save_path=os.path.join(args.data_path, f"{args.dataset_name}/frames/"),
        frame_size=args.frame_size_extractor,
        qscale=args.qscale,
        workers=args.workers,
        original_size=args.original_size
    )
    for csv_path in [train_csv_path, val_csv_path, test_csv_path]:
        if csv_path:
            FrameSampler.run(
                frames_path=os.path.join(args.data_path, f"{args.dataset_name}/frames/"),
                csv_path=csv_path,
                save_path=os.path.join(args.data_path, f"{args.dataset_name}/sampled_split/"),
                frame_size=args.frame_size_sampler,
                only_cpu=args.only_cpu,
                gpu_number=args.gpu_number
            )