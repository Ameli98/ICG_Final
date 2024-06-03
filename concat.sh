#!/bin/bash
echo "file" $1 > VideoList.txt
echo "file" $2 >> VideoList.txt
ffmpeg -f concat -safe 0 -i VideoList.txt -c copy $3
rm VideoList.txt