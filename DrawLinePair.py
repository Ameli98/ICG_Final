import cv2
import numpy as np
from PIL import Image
import csv
import argparse
from time import time

# Initialize global variables
SrcPointBuffer, DstPointBuffer = [], []
t0, t1 = time(), time()

def ColorGenerator(seed):
    np.random.seed(seed)
    return list(np.random.choice(range(256), size=3))

def DrawSrcImage(event, x, y, flags, param):
    global SrcPointBuffer, t0, t1
    if event == cv2.EVENT_LBUTTONDOWN:
        Now = time()
        if Now - t1 > 0.5:
            t0, t1 = t1, Now
        else:
            return
        SrcPointBuffer.extend([y, x])
        print('clicking: ', x, y)
        cv2.circle(param, (x, y), 5, (255, 255, 255), thickness=-1)
        if len(SrcPointBuffer) % 4 == 0:
            Color = ColorGenerator(len(SrcPointBuffer) // 4)
            cv2.line(param, (SrcPointBuffer[-3], SrcPointBuffer[-4]), (SrcPointBuffer[-1], SrcPointBuffer[-2]), 
                     (int(Color[0]), int(Color[1]), int(Color[2])), 2)

def DrawDstImage(event, x, y, flags, param):
    global DstPointBuffer, t0, t1
    if event == cv2.EVENT_LBUTTONDOWN:
        Now = time()
        if Now - t1 > 0.5:
            t0, t1 = t1, Now
        else:
            return
        DstPointBuffer.extend([y, x])
        print('clicking: ', x, y)
        cv2.circle(param, (x, y), 5, (255, 255, 255), thickness=-1)
        if len(DstPointBuffer) % 4 == 0:
            Color = ColorGenerator(len(DstPointBuffer) // 4)
            cv2.line(param, (DstPointBuffer[-3], DstPointBuffer[-4]), (DstPointBuffer[-1], DstPointBuffer[-2]), 
                     (int(Color[0]), int(Color[1]), int(Color[2])), 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="Source Image")
    parser.add_argument("dst", help="Destination Image")
    parser.add_argument("csv", help="Feature Line Pair Data")
    args = parser.parse_args()

    print("Loading images...")
    SrcImage = cv2.imread(args.src)
    DstImage = Image.open(args.dst)
    DstImage = DstImage.resize((SrcImage.shape[1], SrcImage.shape[0]))
    DstImage = cv2.cvtColor(np.array(DstImage), cv2.COLOR_RGB2BGR)

    if SrcImage is None:
        print(f"Error: The source image '{args.src}' could not be loaded.")
        exit()

    if DstImage is None:
        print(f"Error: The destination image '{args.dst}' could not be loaded.")
        exit()

    print("Images loaded successfully.")

    cv2.namedWindow('SrcImage')
    cv2.setMouseCallback('SrcImage', DrawSrcImage, SrcImage)

    cv2.namedWindow('DstImage')
    cv2.setMouseCallback('DstImage', DrawDstImage, DstImage)
    
    while True:
        cv2.imshow('SrcImage', SrcImage)
        cv2.imshow('DstImage', DstImage)
        if cv2.waitKey(20) & 0xFF == 27:  # Escape key to exit
            break
    cv2.destroyAllWindows()

    with open(args.csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Src1y", "Src1x", "Src2y", "Src2x", "Dst1y", "Dst1x", "Dst2y", "Dst2x"])
        writer.writerows(list(zip(SrcPointBuffer[::4], SrcPointBuffer[1::4], SrcPointBuffer[2::4], SrcPointBuffer[3::4], 
                                  DstPointBuffer[::4], DstPointBuffer[1::4], DstPointBuffer[2::4], DstPointBuffer[3::4])))
