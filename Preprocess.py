# Preprocess.py

import cv2
import numpy as np
import math
import Main
import PossibleChar
import DetectChars
import copy
# module level variables ##########################################################################
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5) #####11
ADAPTIVE_THRESH_BLOCK_SIZE = 31
ADAPTIVE_THRESH_WEIGHT = 9






blackL = np.array([0, 0, 0], dtype="uint8")
blackU = np.array([40, 40, 40], dtype="uint8")


whiteL = np.array([40, 40, 40], dtype="uint8")
whiteU = np.array([255, 255, 255], dtype="uint8")


yellowL = np.array([0, 100, 100], dtype="uint8")
yllowU = np.array([100, 255, 255], dtype="uint8")
###################################################################################################
def preprocess(imgOriginal):

    image = imgOriginal

    output2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    imgThresh = cv2.adaptiveThreshold(output2, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)


    return output2, imgThresh


# ###################################################################################################
# def extractValue(imgOriginal):
#     height, width, numChannels = imgOriginal.shape
#
#     imgHSV = np.zeros((height, width, 3), np.uint8)
#
#     imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
#
#     imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
#
#     return imgValue
#
# ###################################################################################################
# def maximizeContrast(imgGrayscale):
#
#     height, width = imgGrayscale.shape
#
#     imgTopHat = np.zeros((height, width, 1), np.uint8)
#     imgBlackHat = np.zeros((height, width, 1), np.uint8)
#
#     structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # Same as np.ones((3,3)
#
#     imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement) # It is difference of  input image and Opening of the image
#     imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement) # it is difference of closing of the input image and input image.
#
#     imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
#     imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
#
#     return imgGrayscalePlusTopHatMinusBlackHat








GAUSSIAN_SMOOTH_FILTER_SIZES = (3, 3) #####11
ADAPTIVE_THRESH_BLOCK_SIZES = 21
ADAPTIVE_THRESH_WEIGHTS = 21

def preprocessS(imgOriginal):


    imgBl = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    imgGrayscale = copy.deepcopy(imgBl)
    '''
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(2, 2))

    lab = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2, a, b))  # merge channels
    im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    '''
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))

    im = clahe.apply(imgBl)
    # im = cv2.GaussianBlur(im, (3,3), 0)


    # im= cv2.fastNlMeansDenoising(imgBl, h=1, templateWindowSize=3, searchWindowSize=3)
    cv2.imshow('plate', im)
    cv2.waitKey(0)
    imgThresh = half_thresh(im)


    # kernel = np.ones((1, 1), np.uint8)
    # imgThresh = cv2.dilate(imgThresh, kernel, iterations=1)


    imgThresh = kray_fill(imgThresh)
    imgForCount = imgThresh.copy()
    h, imgForCount = cv2.threshold(imgForCount, 1, 1, cv2.THRESH_BINARY_INV)

    # find chars
    listOfPossibleChars = []
    noneChar = []
    imgThreshCopy = imgThresh.copy()

            # find all contours in plate
    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in range(len(contours)):                        # for each contour
        possibleChar = PossibleChar.PossibleChar(contours[contour])

        if DetectChars.checkIfPossibleChar(possibleChar):              # if contour is a possible char, note this does not compare to other chars (yet) . . .
            listOfPossibleChars.append(possibleChar)
        else:
            noneChar.append(possibleChar)
    numberOfChars = len(listOfPossibleChars)
    listOfPossibleChars = sorted(listOfPossibleChars, key=lambda obj: obj.intBoundingRectArea)


    listForClear = []

    if numberOfChars > 2:
        for i in noneChar:
            if i.intBoundingRectArea > listOfPossibleChars[-1].intBoundingRectArea and\
                    abs(i.intCenterY - listOfPossibleChars[-2].intCenterY) < 10:
                if listOfPossibleChars[-2].intBoundingRectWidth/i.intBoundingRectWidth > 0.33:
                    listForClear.append(int(i.intCenterX))

                else:
                    step = i.intBoundingRectWidth/3
                    listForClear.append(int(i.intBoundingRectX+step))
                    listForClear.append(int(i.intBoundingRectX+step*2))



        listOfPossibleChars = sorted(listOfPossibleChars, key=lambda obj: obj.intCenterX)
        sm = 0
        for i in range(numberOfChars - 1):
            sm += listOfPossibleChars[i + 1].intCenterX - listOfPossibleChars[i].intCenterX
        meandiff = sm / (numberOfChars - 1)
        if listOfPossibleChars[1].intCenterX - listOfPossibleChars[0].intCenterX > meandiff and abs(listOfPossibleChars[1].intCenterY - listOfPossibleChars[0].intCenterY) > imgThresh.shape[0] / 100 * 10:
            imgThresh[:, :listOfPossibleChars[0].intBoundingRectX + listOfPossibleChars[0].intBoundingRectWidth] = 0
        elif listOfPossibleChars[-1].intCenterX - listOfPossibleChars[-2].intCenterX > meandiff and abs(listOfPossibleChars[-1].intCenterY - listOfPossibleChars[-2].intCenterY) > imgThresh.shape[0] / 100 * 10:
            imgThresh[:, listOfPossibleChars[-1].intBoundingRectX:] = 0



        matchNoneChar = []
        for i in noneChar:
            for j in noneChar:
                if i != j and i not in matchNoneChar and j not in matchNoneChar:
                    if abs(i.intBoundingRectX - j.intBoundingRectX) < imgThresh.shape[1] / 100 * 10 and abs(i.intCenterY - j.intCenterY) > imgThresh.shape[0] / 100 * 10\
                            and j.intBoundingRectArea > 30 and i.intBoundingRectArea > 30 \
                            and j.intBoundingRectWidth > 5 and i.intBoundingRectWidth > 5 \
                            and j.intBoundingRectHeight > 5 and i.intBoundingRectHeight > 5 \
                            and j.intBoundingRectHeight+j.intBoundingRectY < imgThresh.shape[0] - 6 and j.intBoundingRectY > imgThresh.shape[0] / 100 * 10 \
                            and i.intBoundingRectHeight + i.intBoundingRectY < imgThresh.shape[0] - 6   and i.intBoundingRectY > imgThresh.shape[0] / 100 * 10:
                        sm = imgForCount[:, i.intBoundingRectX:i.intBoundingRectX+i.intBoundingRectWidth].sum(axis=0)
                        sm = enumerate(sm, start=i.intBoundingRectX)
                        matchNoneChar.append(i)
                        matchNoneChar.append(j)
                        sm = sorted(sm, key=lambda x: x[1])
                        if i.intBoundingRectWidth >= 4:
                            colForFill = 4
                        else:
                            colForFill = 2
                        for g in range(colForFill):
                            for k in range(imgThresh.shape[0] -1):
                                if imgThresh[k, sm[g][0]] == 255 and imgThresh[k+1, sm[g][0]] == 0:
                                    for l in range(k, imgThresh.shape[0] - 1):
                                        if imgThresh[l+1, sm[g][0]] == 0:
                                            imgThresh[l + 1, sm[g][0]] = 255
                                        else:
                                            break
                                    break

                    # height, width = imgThresh.shape
                    # imgContours = np.zeros((height, width, 3), np.uint8)
                    # cv2.drawContours(imgContours, i.contour, -1, Main.SCALAR_WHITE)
                    # cv2.imshow('kontur', imgContours)
                    # cv2.waitKey(0)


    for i in listForClear:
        imgThresh[:, i] = 0








    imgContours, contours, npaHierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    listOfPossibleChars = []


    for contour in range(len(contours)):                        # for each contour
        possibleChar = PossibleChar.PossibleChar(contours[contour])
        if DetectChars.checkIfPossibleChar(possibleChar):              # if contour is a possible char, note this does not compare to other chars (yet) . . .
            listOfPossibleChars.append(possibleChar)



    imgForRcgnz = imgGrayscale.copy()
    imgForRcgnzCopy = imgForRcgnz.copy()
    for i in listOfPossibleChars:
        roi = imgForRcgnzCopy[:, i.intBoundingRectX -2: i.intBoundingRectX + i.intBoundingRectWidth +2]

        sm = roi.sum()
        area = roi.shape[0] * roi.shape[1]
        h, imgForRcgnz[:, i.intBoundingRectX -2: i.intBoundingRectX + i.intBoundingRectWidth +2] =\
            cv2.threshold(roi, sm / area - 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imshow('tut', imgForRcgnz)
    # cv2.waitKey(0)

    if len(listOfPossibleChars) > 7:
        blue = [np.array([86, 31, 4]), np.array([255, 150, 150])]
        mask = cv2.inRange(imgOriginal, blue[0], blue[1])
        output = cv2.bitwise_and(imgOriginal, imgOriginal, mask=mask)

        # show the images

        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        output = half_thresh(output)
        # cv2.imshow("images", output)
        # cv2.waitKey(0)
        listOfPossibleChars = sorted(listOfPossibleChars, key=lambda x: x.intCenterX)
        frstChar = listOfPossibleChars[0]
        roi = output[frstChar.intBoundingRectY:frstChar.intBoundingRectY+frstChar.intBoundingRectHeight,
              frstChar.intBoundingRectX:frstChar.intBoundingRectX+frstChar.intBoundingRectWidth]
        h, roi = cv2.threshold(roi, 1, 1, cv2.THRESH_BINARY)
        sm = sum(sum(roi))
        area = roi.shape[1] * roi.shape[0]
        if sm/area < 0.3:
            for i in range(frstChar.intBoundingRectX+frstChar.intBoundingRectWidth):
                imgThresh[:, i] = 0
        # cv2.imshow("images", imgThresh)
        # cv2.waitKey(0)
        # for i in listFo i] = 0

    '''    
    imgGrayscale = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    imgBlurred = imgGrayscale.copy()
    # imgBlurred = cv2.GaussianBlur(imgGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZES, 100)  # 2nd parameter is (height,width) of Gaussian kernel,3rd parameter is sigmaX,4th parameter is sigmaY(as not specified it is made same as sigmaX).

    half_ln = imgBlurred.shape[0] / 2 * imgBlurred.shape[1]
    firs_half_img = imgBlurred[:, :int(imgBlurred.shape[1] / 2)]
    second_half_img = imgBlurred[:, int(imgBlurred.shape[1] / 2):]
    firs_sum = firs_half_img.sum()
    second_sum = second_half_img.sum()

    imgForRcgnz = imgBlurred.copy()
    h, imgForRcgnz[:, :int(imgBlurred.shape[1] / 2)] = cv2.threshold(firs_half_img, firs_sum / half_ln - 20, 255,
                                                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, imgForRcgnz[:, int(imgBlurred.shape[1] / 2):] = cv2.threshold(second_half_img, second_sum / half_ln - 20, 255,
                                                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    '''


    # imgBlurred = cv2.GaussianBlur(imgGrayscale, (3, 3), 100)
    # imgForRcgnz = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    kernel = np.ones((3, 3), np.uint8)
    imgForRcgnz = cv2.erode(imgForRcgnz, kernel)
    # cv2.imshow('Erodt', imgForRcgnz)
    # cv2.waitKey(0)

    # imgForRcgnz = imgGrayscale.copy()
    # v = np.median(imgOriginal)
    # sigma = 0.4
    # # ---- apply automatic Canny edge detection using the computed median----
    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) * v))
    # imgThresh = cv2.Canny(imgOriginal, lower, upper)
    # imgThresh = cv2.dilate(imgThresh, (3,3))

    # im = cv2.Canny(img,20,50)

    # _, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # im = np.zeros([img.shape[0], img.shape[1], 3])


    cv2.imshow('Erod', imgThresh)
    cv2.waitKey(0)
    return imgGrayscale, imgThresh, imgForRcgnz


def half_thresh(imgGrayscale):

    imgBlurred = cv2.GaussianBlur(imgGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZES, 100)  # 2nd parameter is (height,width) of Gaussian kernel,3rd parameter is sigmaX,4th parameter is sigmaY(as not specified it is made same as sigmaX).


    half_ln = imgBlurred.shape[0]/2 * imgBlurred.shape[1]
    firs_half_img = imgBlurred[:, :int(imgBlurred.shape[1]/2)]
    second_half_img = imgBlurred[:, int(imgBlurred.shape[1]/2):]
    firs_sum = firs_half_img.sum()
    second_sum = second_half_img.sum()


    imgThresh = imgBlurred.copy()
    h, imgThresh[:, :int(imgBlurred.shape[1]/2)] = cv2.threshold(firs_half_img, firs_sum/half_ln - 255, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    h, imgThresh[:, int(imgBlurred.shape[1]/2):] = cv2.threshold(second_half_img, second_sum/half_ln - 255, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    return imgThresh


def kray_fill(imgThresh):
    imgForCount = imgThresh.copy()
    h, imgForCount = cv2.threshold(imgForCount, 1, 1, cv2.THRESH_BINARY_INV)

    ln = imgThresh.shape[1]
    counterU = 0
    counterB = imgThresh.shape[0] - 1
    counterL = 1
    counterR = imgThresh.shape[1] - 2

    sm = imgForCount.sum(axis=1)
    for i in sm:
        counterU += 1
        if i > int(ln / 100 * 40):
            break

    sm = sm[::-1]
    for i in sm:
        # p
        counterB -= 1
        if i > int(ln / 100 * 70):
            break

    ln = imgThresh.shape[0]
    sm = imgForCount.sum(axis=0)
    for i in sm:
        counterL += 1
        if i > int(ln / 100 * 60):
            break

    sm = sm[::-1]
    for i in sm:
        counterR -= 1
        if i > int(ln / 100 * 60):
            break

    imgThresh[:counterU] = 0
    imgThresh[counterB:] = 0
    imgThresh[:, :counterL] = 0
    imgThresh[:, counterR:] = 0

    return imgThresh