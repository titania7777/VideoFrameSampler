# VideoFrameSampler
Simple and Salient Video Frames Sampling Technique Using the Mean of Deep Features  

## Summary
This code's purpose is to find meaningful frames.  
This repository only provides video frame sampler codes(returns the JSON file), but we will be published training codes later!!.  

## Requirements
*   opencv-python
*   ffmpeg-python
*   torch
*   [pillow-simd(optional)](https://github.com/uploadcare/pillow-simd)

## Usage
Clone this repository
```bash
git clone https://github.com/titania7777/VideoFrameSampler.git
```
Download the dataset
```bash
cd ./VideoFrameSampler/Data/UCF101/
./download.sh
```
Run
```bash
cd ../../
python run.py --dataset-name UCF101 --split-id 1
```

## Examples on ActivityNet
### Kayaking(Uniform)
<img align="center" src="figures/1.PNG" width="750">

### Kayaking(Our)
<img align="center" src="figures/2.PNG" width="750">

### Laying Tile(Uniform)
<img align="center" src="figures/3.PNG" width="750">

### Laying Tile(Our)
<img align="center" src="figures/4.PNG" width="750">