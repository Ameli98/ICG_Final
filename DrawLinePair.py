import cv2
import numpy as np
import random
import csv
import argparse
from time import time


class Point():
    def __init__(self, x:int, y:int) -> None:
        self.Coordinate = (x, y)

    def __getitem__(self, key:int) -> int:
        return self.Coordinate[key]

class Line():
    def __init__(self, Point1:Point, Point2:Point) -> None:
        self.Points = (Point1, Point2)

    def __getitem__(self, key:int) -> Point:
        return self.Points[key]


class LinePairList():
    def __init__(self) -> None:
        self.SrcLineList = []
        self.DstLineList = []
        self.SrcPointBuffer, self.DstPointBuffer = None, None
    
    def __len__(self) -> int:
        return min(len(self.SrcLineList), len(self.DstLineList))
    
    def __getitem__(self, key) -> tuple:
        return self.SrcLineList[key], self.DstLineList[key]

    def SrcAppend(self, p:Point):
        if self.SrcPointBuffer:
            self.SrcLineList.append(Line(self.SrcPointBuffer, p))
            p1 = self.SrcPointBuffer
            self.SrcPointBuffer = None
            return p1, p
        else:
            self.SrcPointBuffer = p
            return None
        
    def DstAppend(self, p:Point):
        if self.DstPointBuffer:
            self.DstLineList.append(Line(self.DstPointBuffer, p))
            p1 = self.DstPointBuffer
            self.DstPointBuffer = None
            return p1, p
        else:
            self.DstPointBuffer = p
            return None
        
    def GetSrcLen(self) -> int:
        return len(self.SrcLineList)
    
    def GetDstLen(self) -> int:
        return len(self.DstLineList)


def ColorGenerator(seed):
    random.seed(seed)
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

def DrawImage(event, x, y, flags, param):
    global LinePairs, t0, t1
    img, src = param[0], param[1]

    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_LBUTTONDBLCLK:
        Now = time()
        if Now - t1 > 0.5:
            t0, t1 = t1, Now
        else:
            return
        Pixel = Point(x, y)
        cv2.circle(img, (x, y), 5, (255, 255, 255), thickness=-1)

        def AddOnce():
            Link = LinePairs.SrcAppend(Pixel) if src else LinePairs.DstAppend(Pixel)
            if Link:
                i = LinePairs.GetSrcLen() if src else LinePairs.GetDstLen()
                Color = ColorGenerator(i)
                cv2.line(img, Link[0].Coordinate, Link[1].Coordinate, (int(Color[0]), int(Color[1]), int(Color[2])), 2)
        AddOnce()
        if flags == cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_LBUTTONDOWN:
            AddOnce()


LinePairs = LinePairList()
t0, t1 = time(), time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="Source Image")
    parser.add_argument("dst", help="Destination Image")
    parser.add_argument("csv", help="Feature Line Pair Data")
    args = parser.parse_args()

    SrcImage = cv2.imread(args.src)
    DstImage = cv2.imread(args.dst)
    DstImage = cv2.resize(DstImage, (SrcImage.shape[1], SrcImage.shape[0]), interpolation=cv2.INTER_CUBIC)

    cv2.namedWindow('SrcImage')
    cv2.setMouseCallback('SrcImage', DrawImage, (SrcImage, True))

    cv2.namedWindow('DstImage')
    cv2.setMouseCallback('DstImage', DrawImage, (DstImage, False))
    
    while True:
        cv2.imshow('SrcImage', SrcImage)
        cv2.imshow('DstImage', DstImage)
        if cv2.waitKey(20) & 0xFF == 27:  # Escape key to exit
            break
    cv2.destroyAllWindows()

    with open(args.csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Srcy1", "Srcx1", "Srcy2", "Srcx2", "Dsty1", "Dstx1", "Dsty2", "Dstx2"])
        for SrcLine, DstLine in LinePairs:
            SrcL1, SrcL2, DstL1, DstL2 = *SrcLine, *DstLine
            SrcX1, SrcY1, SrcX2, SrcY2 = *SrcL1, *SrcL2
            DstX1, DstY1, DstX2, DstY2 = *DstL1, *DstL2
            writer.writerow((SrcY1, SrcX1, SrcY2, SrcX2, DstY1, DstX1, DstY2, DstX2))