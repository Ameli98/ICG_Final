# ColorFusion Morphing
A combination of image morphing and color hamonization.
## Installation
```
pip install numpy opencv-python pandas
```
This program requires SAM model.
Ref : https://github.com/facebookresearch/segment-anything/blob/main/README.md
```
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
Also, ffmpeg is necessary.
## Getting Start
### Draw Feature Pairs
Use left click to draw the feature line
Use shift + left click to snap the point
Use Escape key to quit
```
python3 DrawLinePair.py <Source Image> <Destination Image> <Output csv file>
```

### Image Morphing
```
python3 ImageMorphing.py <Source Image> <Destination Image> <Feature pair csv file> <Output Video>
```

### Color Harmonization
You can choose which parts you want to color harmonize with left click.
```
python3 Harmonization.py <Source Image> <Output Video>
```

### Video Concatenation
```
bash concat.sh <Source Video 1> <Source Video 2> <Output Video>
```
You can create and concatenate the image morphing video and color harmonization video with these code:
```
python3 ImageMorphing.py <Source Image> <Destination Image> <Feature pair csv file> <Output Video 1>
python3 Harmoniztion.py <Destination Image> <Output Video 2>
bash concat.sh <Output Video 1> <Output Video 2> <Concatenated video>
```

## Division of the work
B08504098 : Interface of drawing line pairs, Segmentation, Color harmonization, and Report.
B10902115 : Main structure of Image Morphing and Report.
41173058H : Draw line pairs and Code testing.