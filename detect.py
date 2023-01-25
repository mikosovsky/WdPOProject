import json
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm


def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """
    # Creating kernel matrix
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    # Colors borders in HSV
    lowRedHSV1 = (175, 0, 68)
    highRedHSV1 = (180, 255, 255)
    lowRedHSV2 = (0, 83, 61)
    highRedHSV2 = (4, 255, 255)
    lowGreenHSV = (31, 38, 18)
    highGreenHSV = (90, 255, 255)
    lowYellowHSV = (2, 80, 100)
    highYellowHSV = (33, 255, 255)
    lowPurpleHSV = (115, 25, 0)
    highPurpleHSV = (177, 255, 230)

    # set image sizes for horizontal and vertical images
    verSize = (720, 1280)
    horSize = (1280, 720)
    sqSize = (1280, 1280)
    img = cv2.imread(img_path)
    # Get actual image size
    height, width, channels = img.shape

    # Change size of image to smaller
    if height < width:
        resizedImg = cv2.resize(img, dsize=horSize)
    elif height > width:
        resizedImg = cv2.resize(img, dsize=verSize)
    else:
        resizedImg = cv2.resize(img, dsize=sqSize)

    # Change from BGR to HSV
    resizedImg = cv2.GaussianBlur(resizedImg, (5, 5), 0)
    imgHSV = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2HSV)

    # Geting threshold of every color
    red1Threshold = cv2.inRange(imgHSV, lowRedHSV1, highRedHSV1)
    red2Threshold = cv2.inRange(imgHSV, lowRedHSV2, highRedHSV2)
    redThreshold = red1Threshold + red2Threshold
    redThreshold = cv2.erode(redThreshold, kernel, iterations=1)
    redThreshold = cv2.morphologyEx(redThreshold, cv2.MORPH_OPEN, kernel)
    redThreshold = cv2.morphologyEx(redThreshold, cv2.MORPH_CLOSE, kernel)

    purpleThreshold = cv2.inRange(imgHSV, lowPurpleHSV, highPurpleHSV)
    purpleThreshold = cv2.erode(purpleThreshold, kernel, iterations=1)
    purpleThreshold = cv2.morphologyEx(purpleThreshold, cv2.MORPH_OPEN, kernel)
    purpleThreshold = cv2.morphologyEx(purpleThreshold, cv2.MORPH_CLOSE, kernel)

    greenThreshold = cv2.inRange(imgHSV, lowGreenHSV, highGreenHSV)
    greenThreshold = cv2.erode(greenThreshold, kernel, iterations=1)
    greenThreshold = cv2.morphologyEx(greenThreshold, cv2.MORPH_OPEN, kernel)
    greenThreshold = cv2.morphologyEx(greenThreshold, cv2.MORPH_CLOSE, kernel)

    yellowThreshold = cv2.inRange(imgHSV, lowYellowHSV, highYellowHSV)
    yellowThreshold = cv2.erode(yellowThreshold, kernel, iterations=1)
    yellowThreshold = cv2.morphologyEx(yellowThreshold, cv2.MORPH_OPEN, kernel)
    yellowThreshold = cv2.morphologyEx(yellowThreshold, cv2.MORPH_CLOSE, kernel)

    # Geting contours
    contoursRed, hierarchyRed = cv2.findContours(redThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursGreen, hierarchyGreen = cv2.findContours(greenThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursPurple, hierarchyPurple = cv2.findContours(purpleThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursYellow, hierarchyYellow = cv2.findContours(yellowThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #TODO: Implement detection method.
    
    red = len(contoursRed)
    yellow = len(contoursYellow)
    green = len(contoursGreen)
    purple = len(contoursPurple)

    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
