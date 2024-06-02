import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np

def Segmentation(Image) -> np.array:
    SAM = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth")
    MG = SamAutomaticMaskGenerator(SAM)
    Masks = MG.generate(Image)
    Masks = sorted(Masks, key=(lambda m : m["area"]), reverse=True)
    Label, ColorBlock = np.zeros((Image.shape[0], Image.shape[1])), np.zeros((Image.shape[0], Image.shape[1], 3))
    for i, mask in enumerate(Masks):
        Label[mask["segmentation"]] = i
        ColorBlock[mask["segmentation"]] = np.random.randint(255, size=3)
    
    return Label, ColorBlock

if __name__ == "__main__":
    Image = cv2.imread("./Megumin.jpg")
    Label, ColorBlock = Segmentation(Image)
    ColorBlock = 0.4 * ColorBlock
    PresentImage = cv2.add(Image, ColorBlock.astype(np.uint8))

    cv2.imshow("RGBA", PresentImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()