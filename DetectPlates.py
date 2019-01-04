# DetectPlates.py

import cv2
import numpy as np
import math
import Preprocess
import DetectChars
import PossiblePlate
import PossibleChar

# module level variables ##########################################################################
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5
cascade = cv2.CascadeClassifier('/home/user/PycharmProjects/bot/venv/lib/python3.5/site-packages/cv2/data/haarcascade_russian_plate_number.xml')

###################################################################################################
def detectPlatesInScene(imgOriginalScene):
    listOfPossiblePlates = []                   # this will be the return value

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

        
    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(imgOriginalScene)


    listOfPossibleCharsInScene = findPossibleChars_Plate(imgThreshScene, imgGrayscaleScene) # Here we get a list of all the contours in the image that may be characters.
    # listOfPossibleCharsInScene = validateChars_Plate(imgGrayscaleScene, listOfPossibleCharsInScene)

            # given a list of all possible chars, find groups of matching chars
            # in the next steps each group of matching chars will attempt to be recognized as a plate
    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars_Plate(listOfPossibleCharsInScene, imgGrayscaleScene)

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:                   # for each group of matching chars
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)         # attempt to extract plate

        if possiblePlate.imgPlate is not None:                          # if plate was found
            listOfPossiblePlates.append(possiblePlate)                  # add to list of possible plates

    return listOfPossiblePlates


def findPossibleChars_Plate(imgThresh, gray):
    listOfPossibleChars = []  # this will be the return value

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=3)

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()

    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_CCOMP,
                                                           cv2.CHAIN_APPROX_SIMPLE)  # find all contours

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):  # for each contour

        possibleChar = PossibleChar.PossibleChar(contours[i])  # Here we calculate the x,y,w,h,flatdiagonalsize,aspedctratio,area and (x,y) of the center of the rectangle that is bounding the contour.

        if checkIfPossibleChar_Plate(possibleChar):  # if contour is a possible char, note this does not compare to other chars (yet) . . .
            for (x, y, w, h) in faces:
                if x < possibleChar.intBoundingRectX < x+w and y < possibleChar.intCenterY < y+h:
                    listOfPossibleChars.append(possibleChar)                        # and add to list of possible chars

    return listOfPossibleChars

###################################################################################################
def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()           # this will be the return value

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right based on x position

            # calculate the center point of the plate
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0
    # This is the probable centeral point of this plate.
    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

            # calculate plate width and height
    intPlateWidth = int((listOfMatchingChars[-1].intBoundingRectX + listOfMatchingChars[-1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)
    # Here we calculate the probable width of this plate.
    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars) # Here we calculate the probale height of this particular plate.

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR) # We include the padding factor.

            # calculate correction angle of plate region
    fltOpposite = listOfMatchingChars[-1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[-1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

            # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possiblePlate.rrLocationOfPlateInScene = (tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

            # final steps are to perform the actual rotation

            # get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0) # The first poin tis the point of rotaion or center,theta and scaling factor


    height, width, numChannels = imgOriginal.shape      # unpack original image width and height

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))       # rotate the entire image

    imgTmp = cv2.getRectSubPix(imgRotated, (width, intPlateHeight), tuple(ptPlateCenter))

########################################################################################################################################



    # # im2 = cv2.cvtColor(imgTmp, cv2.COLOR_BGR2GRAY)
    # # im2 = cv2.Laplacian(im2,cv2.CV_8U)
    imgTmp = cv2.cvtColor(imgTmp, cv2.COLOR_BGR2GRAY)
    # # mn = np.percentile(imgTmp, 75)
    # # _, im2 = cv2.threshold(imgTmp, mn, 255, cv2.THRESH_BINARY)
    #
    # v = np.mean(imgTmp)
    # sigma = 0.33
    # # ---- apply automatic Canny edge detection using the computed median----
    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) * v))
    # imgTmp = cv2.Canny(imgTmp, lower, upper)
    gray = cv2.GaussianBlur(imgTmp, (5, 5), 0)
    # gray = cv.equalizeHist(gray)

    # if len(listOfMatchingChars) < 7:
    #     intPlateWidth += listOfMatchingChars[0].intBoundingRectWidth *2
    # if len(listOfMatchingChars) > 8:
    #     intPlateWidth -= listOfMatchingChars[0].intBoundingRectWidth


############################################################################################################################################################
    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter)) # We extract the probable plate from the Original image

    possiblePlate.imgPlate = imgCropped         # copy the cropped plate image into the applicable member variable of the possible plate

    return possiblePlate

















MIN_PIXEL_WIDTH_P = 8
MIN_PIXEL_HEIGHT_P = 10

MIN_ASPECT_RATIO_P = 0.25#######################2.5
MAX_ASPECT_RATIO_P = 1.5#######################1

MIN_PIXEL_AREA_P = 120





def checkIfPossibleChar_Plate(possibleChar):

    # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
    # note that we are not (yet) comparing the char to other chars to look for a group
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA_P and
            possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH_P and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT_P and
            MIN_ASPECT_RATIO_P < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO_P):
        return True
    else:
        return False
