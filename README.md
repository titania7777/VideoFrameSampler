# VideoFrameSampler
Salient Video Frames Sampler for Efficient Model Training Using the Mean of Deep Features  

## Summary
This code's purpose is to find meaningful frames in both trimmed and untrimmed video datasets. And this Sampler working only with UCF101, HMDB51, ActivityNet datasets.  
We only provides video frame sampler codes(returns the JSON file), however, we will be published training codes which utilize this sampler results in another repository later!!.  

## Requirements
*   opencv-python
*   ffmpeg-python
*   torch
*   [pillow-simd(optional)](https://github.com/uploadcare/pillow-simd)

## Usage(UCF101)
Clone this repository
```bash
git clone https://github.com/titania7777/VideoFrameSampler.git
```
Download the dataset
```bash
cd ./VideoFrameSampler/Data/UCF101/
./download.sh
```
Run an Index Sampler
```bash
cd ../../
python sampler_run.py --dataset-name UCF101 --split-id 1
```
Loading Test
```bash
python sampler_test.py --dataset-name UCF101 --split-id 1 --sequence-length 16
```

## Sampled Annotations
We provide our sampler results here
### [Download UCF101 Sampled Annotations](https://www.dropbox.com/s/lue5oeibp2s2r73/UCF101.zip?dl=0)  
### [Download HMDB51 Sampled Annotations](https://www.dropbox.com/s/34v3o8d1ujlqk1h/HMDB51.zip?dl=0)  
### [Download ActivityNet Sampled Annotations](https://www.dropbox.com/s/v87mos9yocsyl3p/ActivityNet.zip?dl=0)  

## Examples on ActivityNet
### Kayaking(Uniform)
<img align="center" src="figures/1.PNG" width="750">

### Kayaking(Our)
<img align="center" src="figures/2.PNG" width="750">

### Laying Tile(Uniform)
<img align="center" src="figures/3.PNG" width="750">

### Laying Tile(Our)
<img align="center" src="figures/4.PNG" width="750">
