import cv2
import numpy as np
import argparse
from Segmentation import Segmentation


def AddLabel(event, x, y, flags, Label):
    global SelectedLabels
    if event == cv2.EVENT_LBUTTONDOWN:
        SelectedLabels.add(int(Label[y, x]))


def Harmonization(Image, Mask):
    HSVImage = cv2.cvtColor(Image, cv2.COLOR_BGR2HSV)
    H = HSVImage[..., 0].astype(np.float16)

    # Find Minimum Sector in the HSV space
    SortedH = np.sort(H[Mask], axis=None)
    Diff = np.zeros_like(SortedH)
    Diff[:-1] = SortedH[1:] - SortedH[:-1]
    Diff[-1] = 256 - (SortedH[-1] - SortedH[0])
    End = np.argmax(Diff)
    Start = 0 if End == len(Diff) else End + 1

    # Mapping Hue
    SectorArc = End - Start if End > Start else 256 + (End - Start)
    if SectorArc < 0:
        SectorArc += 256
        H[H <= End] += 256

    Scale = 32 / SectorArc
    if Scale >= 1:
        return Image
    else:
        Affine = H.mean() * (1 - Scale)

    H[Mask] = Scale * H[Mask] + Affine
    
    HSVImage[..., 0] = H.astype(np.uint8)
    return cv2.cvtColor(HSVImage, cv2.COLOR_HSV2BGR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("InputPath")
    parser.add_argument("OutputPath")
    args = parser.parse_args()

    InputImage = cv2.imread(args.InputPath)
    Label, ColorBlock = Segmentation(InputImage)
    ColorBlock = 0.4 * ColorBlock
    PresentImage = cv2.add(InputImage, ColorBlock.astype(np.uint8))

    SelectedLabels = set()

    cv2.namedWindow("SelectPart")
    cv2.setMouseCallback("SelectPart", AddLabel, Label)
    while True:
        cv2.imshow("SelectPart", PresentImage)
        if cv2.waitKey(20) & 0xFF == 27:  # Escape key to exit
            break
    cv2.destroyAllWindows()

    # TODO: With selected labels, fetch these pixels for color hamonization
    Mask = np.full((InputImage.shape[0], InputImage.shape[1]), False)
    for l in SelectedLabels:
        Mask[Label == l] = True

    OutputImage = Harmonization(InputImage, Mask)
    cv2.imwrite(args.OutputPath, OutputImage)