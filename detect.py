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
    #Colors borders in HSV
    lowRedHSV1 = (175, 61, 68)
    highRedHSV1 = (360, 255, 255)
    lowRedHSV2 = (0, 83, 61)
    highRedHSV2 = (4, 255, 255)
    lowGreenHSV = (31, 38, 18)
    highGreenHSV = (90, 255, 255)
    lowYellowHSV = (2, 110, 177)
    highYellowHSV = (33, 255, 255)
    lowPurpleHSV = (130, 48, 33)
    highPurpleHSV = (164, 199, 255)

    # set image sizes for horizontal and vertical images
    verSize = (720, 1280)
    horSize = (1280, 720)
    sqSize = (1280, 1280)

    #Load thie image from file
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    #Change from BGR to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Get actual image size
    height, width, channels = img.shape

    #Change size of image to smaller
    if height < width:
        resizedImg = cv2.resize(imgHSV, dsize=horSize)
    elif height > width:
        resizedImg = cv2.resize(imgHSV, dsize=verSize)
    else:
        resizedImg = cv2.resize(imgHSV, dsize=sqSize)


    #TODO: Implement detection method.
    
    red = 0
    yellow = 0
    green = 0
    purple = 0

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
