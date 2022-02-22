import cv2


def select_rois(images: list, downscale: int = 4) -> list:
    rois = []
    for image in images:
        image_small = cv2.resize(image, (int(image.shape[1] / downscale), int(image.shape[0] / downscale)))
        r = cv2.selectROI(image_small)
        roi = [int(r[1] * 4), int((r[1] + r[3]) * 4), int(r[0] * 4), int((r[0] + r[2]) * 4)]
        rois.append(roi)

    cv2.destroyAllWindows()
    return rois