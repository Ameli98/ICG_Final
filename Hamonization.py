import cv2
import numpy as np
import argparse
from Segmentation import Segmentation


def AddLabel(event, x, y, flags, Label):
    global SelectedLabels
    if event == cv2.EVENT_LBUTTONDOWN:
        SelectedLabels.add(int(Label[y, x]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("InputPath")
    # parser.add_argument("OutputPath")
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