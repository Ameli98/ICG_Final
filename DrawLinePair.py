import cv2
import numpy as np
import csv
import argparse

SrcPointBuffer, DstPointBuffer = [], []

def ColorGenerator(seed):
    np.random.seed(seed)
    return list(np.random.choice(range(256), size=3))

def DrawSrcImage(event, x, y, flags, param):
    global SrcPointBuffer
    if event == cv2.EVENT_LBUTTONDOWN:
        SrcPointBuffer.append((x, y))
        print('clicking: ', x, y)
        cv2.circle(param, (x, y), 5, (255, 255, 255), thickness=-1)  # Slightly bigger circle for visibility
        if len(SrcPointBuffer) % 2 == 0:
            Color = ColorGenerator(len(SrcPointBuffer) // 2)
            cv2.line(param, SrcPointBuffer[-2], SrcPointBuffer[-1], (int(Color[0]), int(Color[1]), int(Color[2])), 2)

def DrawDstImage(event, x, y, flags, param):
    global DstPointBuffer
    if event == cv2.EVENT_LBUTTONDOWN:
        DstPointBuffer.append((x, y))
        print('clicking: ', x, y)
        cv2.circle(param, (x, y), 5, (255, 255, 255), thickness=-1)  # Slightly bigger circle for visibility
        if len(DstPointBuffer) % 2 == 0:
            Color = ColorGenerator(len(DstPointBuffer) // 2)
            cv2.line(param, DstPointBuffer[-2], DstPointBuffer[-1], (int(Color[0]), int(Color[1]), int(Color[2])), 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="Source Image")
    parser.add_argument("dst", help="Destination Image")
    parser.add_argument("csv", help="Feature Line Pair Data")
    args = parser.parse_args()

    SrcImage = cv2.imread(args.src)
    DstImage = cv2.imread(args.dst)
    if SrcImage is None or DstImage is None:
        print("Error: One of the images could not be loaded.")
        exit()

    cv2.namedWindow('SrcImage')
    cv2.setMouseCallback('SrcImage', DrawSrcImage, SrcImage)

    cv2.namedWindow('DstImage')
    cv2.setMouseCallback('DstImage', DrawDstImage, DstImage)
    
    while(True):
        cv2.imshow('SrcImage', SrcImage)
        cv2.imshow('DstImage', DstImage)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

    with open(args.csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Src1", "Src2", "Dst1", "Dst2"])
        writer.writerows(list(zip(SrcPointBuffer[::2],SrcPointBuffer[1::2], DstPointBuffer[::2], DstPointBuffer[1::2])))