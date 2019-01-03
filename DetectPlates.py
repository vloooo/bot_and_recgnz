# DetectPlates.py

import cv2
import numpy as np
import math
import Main
import random
import Preprocess
import DetectChars
from PIL import Image
import PossiblePlate
import PossibleChar

# module level variables ##########################################################################
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5


###################################################################################################
def detectPlatesInScene(imgOriginalScene):
    listOfPossiblePlates = []  # this will be the return value

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    if Main.showSteps == True:  # show steps #######################################################
        # cv2.imshow("0", imgOriginalScene)
        Image.fromarray(imgOriginalScene).show()
        input('Press any key to continue...')

    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(
        imgOriginalScene)  # preprocess to get grayscale and threshold images
    # cv2.imshow('laps', imgThreshScene)
    # cv2.waitKey(0)
    # find all possible chars in the scene,
    # this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
    listOfPossibleCharsInScene = findPossibleChars_Plate(
        imgThreshScene)  # Here we get a list of all the contours in the image that may be characters.
    # listOfPossibleCharsInScene = validateChars_Plate(imgGrayscaleScene, listOfPossibleCharsInScene)
    '''
    if True:  # show steps #######################################################
        # print("step 2 - len(listOfPossibleCharsInScene) = " + str(len(listOfPossibleCharsInScene)))

        imgContours = np.zeros((height, width, 3), np.uint8)

        contours = []

        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)

        cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
        cv2.imshow('ll', imgContours)
        cv2.waitKey(0)
    '''
    # This is for the boxing of all the contours
    """
        for possibleChar in listOfPossibleCharsInScene:
            cv2.rectangle(imgContours,(possibleChar.intBoundingRectX,possibleChar.intBoundingRectY),(possibleChar.intBoundingRectX+possibleChar.intBoundingRectWidth,possibleChar.intBoundingRectY+possibleChar.intBoundingRectHeight),(0.0, 255.0, 255.0),1)
            cv2.imshow('PossiblePlate',imgContours)
            cv2.waitKey(0)
    """

    # given a list of all possible chars, find groups of matching chars
    # in the next steps each group of matching chars will attempt to be recognized as a plate
    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene,
                                                                                   imgGrayscaleScene)
    if Main.showSteps == True:  # show steps #######################################################
        print("step 3 - listOfListsOfMatchingCharsInScene.Count = " + str(
            len(listOfListsOfMatchingCharsInScene)))  # 13 with MCLRNF1 image

        imgContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            # imgContours2 = np.zeros((height, width, 3), np.uint8)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
            # end for

            # cv2.drawContours(imgContours, contours, -1, (255, 255, 255))
            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
        # end for

        # imgContours = Image.fromarray(imgContours,'RGB').show()

    # end if # show steps #########################################################################
    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:  # for each group of matching chars
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)  # attempt to extract plate

        if possiblePlate.imgPlate is not None:  # if plate was found
            listOfPossiblePlates.append(possiblePlate)  # add to list of possible plates

    if Main.showSteps == True:
        print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")
    if Main.showSteps == True:  # show steps #######################################################
        print("\n")

        Image.fromarray(imgContours, 'RGB').show()
        input('Press any key to continue...')
        for i in range(0, len(listOfPossiblePlates)):
            p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)

            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), Main.SCALAR_RED, 2)

            # cv2.imshow("4a", imgContours)

            print("possible plate " + str(i) + ", click on any image and press a key to continue . . .")
            # Image.fromarray(listOfPossiblePlates[i].imgPlate,'RGB').show()

        # end for
        print("\nplate detection complete, press a key to begin char recognition . . .\n")
        input()
    # end if # show steps #########################################################################

    return listOfPossiblePlates


###################################################################################################
def findPossibleChars_Plate(imgThresh):
    listOfPossibleChars = []  # this will be the return value

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()
    # print('Now we start to find the contours in the thresholded image that may be characters:')

    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_CCOMP,
                                                           cv2.CHAIN_APPROX_SIMPLE)  # find all contours

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):  # for each contour

        if Main.showSteps == True:  # show steps ###################################################
            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_YELLOW)
            # Image.fromarray(imgContours,'RGB').show()

        possibleChar = PossibleChar.PossibleChar(contours[
                                                     i])  # Here we calculate the x,y,w,h,flatdiagonalsize,aspedctratio,area and (x,y) of the center of the rectangle that is bounding the contour.

        if checkIfPossibleChar_Plate(
                possibleChar):  # if contour is a possible char, note this does not compare to other chars (yet) . . .

            intCountOfPossibleChars = intCountOfPossibleChars + 1  # increment count of possible chars
            listOfPossibleChars.append(possibleChar)  # and add to list of possible chars
            cv2.drawContours(imgContours, contours, i, Main.SCALAR_WHITE)
            # print('This contour may be a character :')
        # else:
        # print('This contour is not a character :')
        # end if
    # end for

    if Main.showSteps == True:  # show steps #######################################################
        print("\nstep 2 - Total number of contours found in the image are = " + str(len(contours)))
        print("step 2 - number of contours those may be characters = " + str(intCountOfPossibleChars))
        # print("These are the contours those may be characters :")
        Image.fromarray(imgContours, 'RGB').show()
    # end if # show steps #########################################################################

    return listOfPossibleChars


###################################################################################################
def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()  # this will be the return value

    listOfMatchingChars.sort(
        key=lambda matchingChar: matchingChar.intCenterX)  # sort chars from left to right based on x position

    # calculate the center point of the plate
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[
        len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[
        len(listOfMatchingChars) - 1].intCenterY) / 2.0
    # This is the probable centeral point of this plate.
    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    # calculate plate width and height
    intPlateWidth = int((listOfMatchingChars[-1].intBoundingRectX + listOfMatchingChars[-1].intBoundingRectWidth -
                         listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)
    # Here we calculate the probable width of this plate.
    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight

    fltAverageCharHeight = intTotalOfCharHeights / len(
        listOfMatchingChars)  # Here we calculate the probale height of this particular plate.

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)  # We include the padding factor.

    # calculate correction angle of plate region
    fltOpposite = listOfMatchingChars[-1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[-1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possiblePlate.rrLocationOfPlateInScene = (
    tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg)

    # final steps are to perform the actual rotation

    # get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg,
                                             1.0)  # The first poin tis the point of rotaion or center,theta and scaling factor

    height, width, numChannels = imgOriginal.shape  # unpack original image width and height

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))  # rotate the entire image

    imgTmp = cv2.getRectSubPix(imgRotated, (width, intPlateHeight), tuple(ptPlateCenter))

    ########################################################################################################################################

    # # im2 = cv2.cvtColor(imgTmp, cv2.COLOR_BGR2GRAY)
    # # im2 = cv2.Laplacian(im2,cv2.CV_8U)
    imgTmp = cv2.cvtColor(imgTmp, cv2.COLOR_BGR2GRAY)
    # # mn = np.percentile(imgTmp, 75)
    # # _, im2 = cv2.threshold(imgTmp, mn, 255, cv2.THRESH_BINARY)
    #
    v = np.mean(imgTmp)
    sigma = 0.33
    # ---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    imgTmp = cv2.Canny(imgTmp, lower, upper)

    # cv2.imshow('llllllllooo', imgTmp)
    # cv2.waitKey(0)
    # if len(listOfMatchingChars) < 7:
    #     intPlateWidth += listOfMatchingChars[0].intBoundingRectWidth *2
    # if len(listOfMatchingChars) > 8:
    #     intPlateWidth -= listOfMatchingChars[0].intBoundingRectWidth

    ############################################################################################################################################################
    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight),
                                   tuple(ptPlateCenter))  # We extract the probable plate from the Original image

    possiblePlate.imgPlate = imgCropped  # copy the cropped plate image into the applicable member variable of the possible plate

    return possiblePlate


MIN_PIXEL_WIDTH = 8
MIN_PIXEL_HEIGHT = 10

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 120


def checkIfPossibleChar_Plate(possibleChar):
    # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
    # note that we are not (yet) comparing the char to other chars to look for a group
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
            possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
            MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False


def validateChars_Plate(img, listOfChars):
    full_contours = []

    v = np.median(img)
    sigma = 0.33
    # ---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    im = cv2.Canny(img, lower, upper)

    _, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    im = np.zeros([img.shape[0], img.shape[1], 3])
    listToVld = []
    # cv2.drawContours(im, contours, -1, Main.SCALAR_WHITE)

    for i in range(len(contours)):  # for each contour
        # cv2.drawContours(im, contours, i, Main.SCALAR_WHITE)
        # cv2.imshow('lap2', im)
        # cv2.waitKey(0)
        possibleChar = PossibleChar.PossibleChar(contours[i])  # Here we calcula
        if checkIfPossibleChar_Plate(possibleChar):
            for k in listOfChars:
                if abs(k.intCenterX - possibleChar.intCenterX) <= 10 and abs(
                        k.intCenterY - possibleChar.intCenterY) <= 10:
                    listToVld.append(k)
                    # break
                    cv2.drawContours(im, contours, i, Main.SCALAR_YELLOW)

    # cv2.imshow('lap22', im)
    # cv2.waitKey(0)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    v = np.median(img)
    sigma = 0.33
    # ---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    im = cv2.Canny(img, lower, upper)

    _, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    im = np.zeros([img.shape[0], img.shape[1], 3])
    # print(len(contours))
    # cv2.drawContours(im, contours, -1, (0, 255, 0), 1)
    cv2.drawContours(im, contours, -1, Main.SCALAR_WHITE)

    for i in range(len(contours)):  # for each contour

        possibleChar = PossibleChar.PossibleChar(contours[i])  # Here we calcula
        if checkIfPossibleChar_Plate(possibleChar):

            for k in listOfChars:
                if abs(k.intCenterX - possibleChar.intCenterX) <= 10 and abs(
                        k.intCenterY - possibleChar.intCenterY) <= 10:
                    listToVld.append(k)
                    # break
                    cv2.drawContours(im, contours, i, Main.SCALAR_YELLOW)

    # cv2.imshow('lap2', im)
    # cv2.waitKey(0)
    return listToVld