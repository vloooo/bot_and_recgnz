# Preprocess.py

import cv2
import numpy as np
import math
import Main
import PossibleChar
import DetectChars
import copy
import matplotlib.pyplot as plt
# module level variables ##########################################################################
import PossiblePlate

PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5

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
    # output2 = cv2.blur(output2, GAUSSIAN_SMOOTH_FILTER_SIZE)

    imgThresh = cv2.adaptiveThreshold(output2, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)


    return output2, imgThresh





GAUSSIAN_SMOOTH_FILTER_SIZES = (3, 3) #####11
ADAPTIVE_THRESH_BLOCK_SIZES = 21
ADAPTIVE_THRESH_WEIGHTS = 21

def preprocessS(imgOriginal):


    imgBl = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    imgGrayscale = copy.deepcopy(imgBl)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))

    im = clahe.apply(imgBl)
    # im = cv2.GaussianBlur(im, (3,3), 0)


    # im= cv2.fastNlMeansDenoising(imgBl, h=1, templateWindowSize=3, searchWindowSize=3)
    # cv2.imshow('plate', im)
    # cv2.waitKey(0)


    imgThresh = half_thresh(im)
    imgThresh = kray_fill_new(imgThresh)

    listOfPossibleChars, numberOfChars, noneChar = find_psbl_chr(imgThresh)
    if numberOfChars > 2:
        imgThresh = separate_stick(imgThresh, listOfPossibleChars, noneChar)
    listOfPossibleChars, _, _ = find_psbl_chr(imgThresh)



    # croped = False
    # if len(listOfPossibleChars) > 7 or len(listOfPossibleChars) == 0:
    imgThresh, imgGrayscale = new_kray_fill(im, imgGrayscale, listOfPossibleChars)
    croped = True
    #     # print('jhfewkhwefohvddcdd')
    # else:
    #     imgThresh = im.copy()
    imgThresh = half_thresh(imgThresh)
    imgThresh = kray_fill(imgThresh)


    imgForCount = imgThresh.copy()
    h, imgForCount = cv2.threshold(imgForCount, 1, 1, cv2.THRESH_BINARY_INV)

    listOfPossibleChars, _, _ = find_psbl_chr(imgThresh)

    # while True:
    if len(listOfPossibleChars) > 7:
        imgThresh = replace_last_first(imgThresh, listOfPossibleChars, len(listOfPossibleChars))

        imgThresh = del_blue(imgOriginal, listOfPossibleChars, imgThresh)


    listOfPossibleChars, _, _ = find_psbl_chr(imgThresh)


    imgThresh = draw_top_line(listOfPossibleChars, imgThresh)
    imgThresh = draw_btm_line(listOfPossibleChars, imgThresh)

    listOfPossibleChars, numberOfChars, noneChar = find_psbl_chr(imgThresh)

    if numberOfChars >= 2:
        imgThresh = separate_stick(imgThresh, listOfPossibleChars, noneChar)


    listOfPossibleChars, numberOfChars, noneChar = find_psbl_chr_4broken(imgThresh)

    imgThresh = matching_broken_chars(imgThresh, noneChar, imgForCount)







    listOfPossibleChars, _, _ = find_psbl_chr(imgThresh)



    height, width = imgThresh.shape
    imgContoursd = np.zeros((height, width, 3), np.uint8)
    lst = []
    for i in listOfPossibleChars:
        lst.append(i.contour)
    # cv2.drawContours(imgContoursd, lst, -1, Main.SCALAR_WHITE)
    # cv2.imshow('konturh', imgContoursd)
    # # cv2.imshow('kontur2h', imgContours)
    #
    # cv2.waitKey(0)



    listOfPossibleChars, _, _ = find_psbl_chr(imgThresh)
    height, width = imgThresh.shape
    imgContours = np.zeros((height, width), np.uint8)
    imgForRcgnz = np.zeros((height, width), np.uint8)

    imgForRcgnzCopy = imgGrayscale.copy()
    for i in listOfPossibleChars:
        x1 = np.max([i.intBoundingRectX - 2, 0])
        y1 = np.min([i.intBoundingRectY + i.intBoundingRectHeight + 1, height-1])
        roi = imgForRcgnzCopy[:, x1: i.intBoundingRectX + i.intBoundingRectWidth + 2]

        h, imgForRcgnz[i.intBoundingRectY - 1:y1, x1: i.intBoundingRectX + i.intBoundingRectWidth + 2] = \
            cv2.threshold(roi[i.intBoundingRectY - 1:y1, :], 2, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h, imgContours[i.intBoundingRectY - 1:y1,
           x1: i.intBoundingRectX + i.intBoundingRectWidth + 2] = \
            cv2.threshold(roi[i.intBoundingRectY - 1:y1, :], 2, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)



    listOfPossibleChars, numberOfChars, noneChar = find_psbl_chr(imgContours)
    if numberOfChars > 0 and croped:
        imgContours = rotate(imgContours, listOfPossibleChars)
        imgForRcgnz = rotate(imgForRcgnz, listOfPossibleChars)

    _, imgContours = cv2.threshold(imgContours, 200, 255, cv2.THRESH_BINARY)
    listOfPossibleChars, numberOfChars, noneChar = find_psbl_chr(imgContours)
    #
    imgContours = draw_top_line(listOfPossibleChars, imgContours)
    imgContours = draw_btm_line(listOfPossibleChars, imgContours)

    listOfPossibleChars, numberOfChars, noneChar = find_psbl_chr_4broken(imgContours)



    if numberOfChars >= 2:
        imgContours = separate_stick(imgContours, listOfPossibleChars, noneChar)

    listOfPossibleChars, numberOfChars, noneChar = find_psbl_chr_4broken(imgContours)
    imgContours = matching_broken_chars(imgContours, noneChar, imgForCount)

    kernel = np.ones((3, 3), np.uint8)
    imgForRcgnz = cv2.erode(imgForRcgnz, kernel)
    # imgContours =
    # cv2.imshow('Erod', imgContours)
    # cv2.waitKey(0)
    return imgGrayscale, imgContours, imgForRcgnz


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
    counterL = 0
    counterR = imgThresh.shape[1] - 1

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


def kray_fill_new(imgThresh):
    imgForCount = imgThresh.copy()
    h, imgForCount = cv2.threshold(imgForCount, 1, 1, cv2.THRESH_BINARY_INV)

    ln = imgThresh.shape[1]
    counterU = 0
    counterB = imgThresh.shape[0] - 1
    counterL = 1
    counterR = imgThresh.shape[1] - 1

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

    return imgThresh

def draw_top_line(listOfPossibleChars,imgThresh):
    tops = [x.intBoundingRectY for x in listOfPossibleChars]
    try:
        topClearLine = int(sum(tops)/len(tops))
        imgThresh[topClearLine, :] = 0
    except ZeroDivisionError:
        print('', end='')
    return imgThresh



def draw_btm_line(listOfPossibleChars, imgThresh):
    if len(listOfPossibleChars):
        btms = [x.intBoundingRectY + x.intBoundingRectHeight for x in listOfPossibleChars]
        # print(imgThresh.shape[0])
        btmClearLine = np.min([int(np.mean(btms)) + 1, imgThresh.shape[0] - 1])
        imgThresh[btmClearLine, :] = 0

    return imgThresh


def find_psbl_chr(imgThresh):
    listOfPossibleChars = []
    noneChar = []
    imgThreshCopy = imgThresh.copy()

            # find all contours in plate
    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in range(len(contours)):                        # for each contour
        possibleChar = PossibleChar.PossibleChar(contours[contour])

        if checkIfPossibleChar_PreProc(possibleChar):              # if contour is a possible char, note this does not compare to other chars (yet) . . .
            listOfPossibleChars.append(possibleChar)
        else:
            noneChar.append(possibleChar)
    numberOfChars = len(listOfPossibleChars)
    listOfPossibleChars = sorted(listOfPossibleChars, key=lambda obj: obj.intBoundingRectArea)
    return listOfPossibleChars, numberOfChars, noneChar





def find_psbl_chr_4broken(imgThresh):
    listOfPossibleChars = []
    noneChar = []
    imgThreshCopy = imgThresh.copy()

            # find all contours in plate
    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in range(len(contours)):                        # for each contour
        possibleChar = PossibleChar.PossibleChar(contours[contour])

        if checkIfPossibleChar_Br(possibleChar):              # if contour is a possible char, note this does not compare to other chars (yet) . . .
            listOfPossibleChars.append(possibleChar)
        else:
            noneChar.append(possibleChar)
    numberOfChars = len(listOfPossibleChars)
    listOfPossibleChars = sorted(listOfPossibleChars, key=lambda obj: obj.intBoundingRectArea)
    return listOfPossibleChars, numberOfChars, noneChar


def matching_broken_chars(imgThresh, noneChar, imgForCount):
    matchNoneChar = []
    for i in noneChar:
        for j in noneChar:
            if i != j and i not in matchNoneChar and j not in matchNoneChar:
                if abs(i.intBoundingRectX - j.intBoundingRectX) < imgThresh.shape[1] / 100 * 10 and abs(
                        i.intCenterY - j.intCenterY) > imgThresh.shape[0] / 100 * 10 \
                        and j.intBoundingRectArea > 30 and i.intBoundingRectArea > 30 \
                        and j.intBoundingRectWidth > 5 and i.intBoundingRectWidth > 5 \
                        and j.intBoundingRectHeight > 5 and i.intBoundingRectHeight > 5 \
                        and j.intBoundingRectHeight + j.intBoundingRectY < imgThresh.shape[
                    0] - 6 and j.intBoundingRectY > imgThresh.shape[0] / 100 * 10 \
                        and i.intBoundingRectHeight + i.intBoundingRectY < imgThresh.shape[
                    0] - 6 and i.intBoundingRectY > imgThresh.shape[0] / 100 * 10:
                    sm = imgForCount[:, i.intBoundingRectX:i.intBoundingRectX + i.intBoundingRectWidth].sum(axis=0)
                    sm = enumerate(sm, start=i.intBoundingRectX)
                    matchNoneChar.append(i)
                    matchNoneChar.append(j)
                    sm = sorted(sm, key=lambda x: x[1])
                    if i.intBoundingRectWidth >= 4:
                        colForFill = 4
                    else:
                        colForFill = 2
                    for g in range(colForFill):
                        for k in range(imgThresh.shape[0] - 1):
                            if imgThresh[k, sm[g][0]] == 255 and imgThresh[k + 1, sm[g][0]] == 0:
                                for l in range(k, imgThresh.shape[0] - 1):
                                    if imgThresh[l + 1, sm[g][0]] == 0:
                                        imgThresh[l + 1, sm[g][0]] = 255
                                    else:
                                        break
                                break

                # height, width = imgThresh.shape
                # imgContours = np.zeros((height, width, 3), np.uint8)
                # cv2.drawContours(imgContours, i.contour, -1, Main.SCALAR_WHITE)
                # cv2.imshow('kontur', imgContours)
                # cv2.waitKey(0)
    return imgThresh


def replace_last_first(imgThresh, listOfPossibleChars, numberOfChars):
    listOfPossibleChars = sorted(listOfPossibleChars, key=lambda obj: obj.intCenterX)
    sm = 0
    numberOfChars = len(listOfPossibleChars)
    for i in range(1, numberOfChars - 1):
        sm += listOfPossibleChars[i + 1].intBoundingRectX - (listOfPossibleChars[i].intBoundingRectX + listOfPossibleChars[i].intBoundingRectWidth)
    meandiff = sm / (numberOfChars - 1)
    # print(meandiff, abs(listOfPossibleChars[1].intBoundingRectX - (listOfPossibleChars[0].intBoundingRectX + listOfPossibleChars[0].intBoundingRectWidth)))
    if abs(listOfPossibleChars[1].intBoundingRectX - (listOfPossibleChars[0].intBoundingRectX + listOfPossibleChars[0].intBoundingRectWidth)) > meandiff or abs(
            listOfPossibleChars[1].intBoundingRectY - listOfPossibleChars[0].intBoundingRectY) > imgThresh.shape[0] / 100 * 3 or abs(
        (listOfPossibleChars[1].intBoundingRectY + listOfPossibleChars[1].intBoundingRectHeight) -
        (listOfPossibleChars[0].intBoundingRectY + listOfPossibleChars[0].intBoundingRectHeight)) > imgThresh.shape[0] / 100 * 3:
        imgThresh[:, :listOfPossibleChars[0].intBoundingRectX + listOfPossibleChars[0].intBoundingRectWidth] = 0
    if listOfPossibleChars[-1].intCenterX - listOfPossibleChars[-2].intCenterX > meandiff and abs(
            listOfPossibleChars[-1].intCenterY - listOfPossibleChars[-2].intCenterY) > imgThresh.shape[0] / 100 * 10:
        imgThresh[:, listOfPossibleChars[-1].intBoundingRectX:] = 0
    return imgThresh


def separate_stick(imgThresh, listOfPossibleChars, noneChar):
    listForClear = []


    for i in noneChar:
        if i.intBoundingRectArea > listOfPossibleChars[-1].intBoundingRectArea*1.5 and \
                abs(i.intCenterY - listOfPossibleChars[-2].intCenterY) < 10 and \
            i.intBoundingRectHeight > 10:

            # print('dsgksforf')
            if listOfPossibleChars[-2].intBoundingRectWidth / i.intBoundingRectWidth > 0.33:
                listForClear.append(int(i.intCenterX))

            elif listOfPossibleChars[-2].intBoundingRectWidth / i.intBoundingRectWidth > 0.25:
                step = i.intBoundingRectWidth / 3
                listForClear.append(int(i.intBoundingRectX + step))
                listForClear.append(int(i.intBoundingRectX + step * 2))

            else:
                step = i.intBoundingRectWidth / 4
                listForClear.append(int(i.intBoundingRectX + step))
                listForClear.append(int(i.intBoundingRectX + step * 2))
                listForClear.append(int(i.intBoundingRectX + step * 3))



    for i in listForClear:
        imgThresh[:, i] = 0
        # cv2.imshow('dj', imgThresh)
        # cv2.waitKey(0)

    return imgThresh


def del_blue(imgOriginal, listOfPossibleChars, imgThresh):
    blue = [np.array([86, 31, 4]), np.array([255, 150, 150])]
    mask = cv2.inRange(imgOriginal, blue[0], blue[1])
    output = cv2.bitwise_and(imgOriginal, imgOriginal, mask=mask)

    # show the images
    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    # output = half_thresh(output)
    d, output = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    listOfPossibleChars = sorted(listOfPossibleChars, key=lambda x: x.intCenterX)
    frstChar = listOfPossibleChars[0]
    roi = output[frstChar.intBoundingRectY:frstChar.intBoundingRectY + frstChar.intBoundingRectHeight,
          frstChar.intBoundingRectX:frstChar.intBoundingRectX + frstChar.intBoundingRectWidth]
    # cv2.imshow("images", roi)
    # cv2.waitKey(0)

    h, roi = cv2.threshold(roi, 1, 1, cv2.THRESH_BINARY)

    # print(roi)
    sm = np.sum(roi)
    area = roi.shape[1] * roi.shape[0]
    if sm / area < 0.3:
        for i in range(frstChar.intBoundingRectX + frstChar.intBoundingRectWidth):
            imgThresh[:, i] = 0
    # cv2.imshow("images", imgThresh)
    # cv2.waitKey(0)
    # for i in listFo i] = 0
    return imgThresh

def new_kray_fill(imgThresh, imgGrayScale, listOfPossibleChars):






    im2 = imgThresh
    width, height = im2.shape
    hist = []

    for x in range(height -1):
        column = im2[0:height -1, x:x+1]
        hist.append(np.sum(column) / len(column))



    hist = np.array(hist)
    # border = np.percentile(hist[10:-11], 75)
    border = np.mean(hist[10:-11]) + 5

    # print(hist)
    frame = 5
    indx = 0
    indx2 = 0
    for x in range(int(height/frame) -1):
        mn = np.amin(hist[x*frame +1:(x+1)*frame +1])
        new_mn = np.amin(hist[(x+1)*frame +1:(x+2)*frame +1])
        # print(hist[x*frame:(x+1)*frame], mn, hist[(x+1)*frame:(x+2)*frame], new_mn)

        if mn < new_mn:
            indx = np.argmin(hist[x*frame +1:x*frame+frame +1]) + x*frame +1
            # imgThresh[:, :indx] = 0
            break







    frame = 5
    first_max = np.max(hist[:11])
    for x in range(int(height/frame)):
        mx = np.amax(hist[x*frame:(x+1)*frame])
        new_mx = np.amax(hist[(x+1)*frame:(x+2)*frame])
        indx2 = np.argmax(hist[x * frame:(x+1) * frame ]) + x * frame

        if mx > new_mx and ((mx > border and indx2>indx) or (mx > first_max +10 and indx2<indx)):
            indx2 = np.argmax(hist[x*frame:x*frame+frame]) + x*frame


            for i, kd in zip(listOfPossibleChars, range(len(listOfPossibleChars))):
                if indx2 < i.intCenterX and indx2 > i. intBoundingRectX:
                    indx2 = i.intBoundingRectX - 2
                    break
            if indx2 < 0:
                indx2 = 1

            imgThresh = imgThresh[:, indx2:]
            imgGrayScale = imgGrayScale[:, indx2:]
            break
    shift = indx2


    # plt.plot(hist)
    # plt.plot([0, height], [border, border])
    # plt.plot([indx, indx], [np.max(hist), np.min(hist)], c='r')
    #
    # plt.plot([indx2, indx2], [np.max(hist), np.min(hist)])
    # #
    # #
    # plt.show()







############################################################################################

    im2 = imgThresh
    width, height = im2.shape
    hist = []

    for x in range(height -1):
        column = im2[0:height -1, x:x+1]
        hist.append(np.sum(column) / len(column))



    hist = np.array(hist)


    frame = 5
    indx = 0
    indx2 = 0
    for x in range(int(height/frame), 1, -1):
        mn = np.amin(hist[x*frame-frame:x*frame])
        new_mn = np.amin(hist[(x-1)*frame-frame:(x-1)*frame])
        if mn < new_mn:
            indx = np.argmin(hist[x*frame-frame:x*frame]) + x*frame-frame
            break



    frame = 5
    first_max = np.max(hist[-12:])
    for x in range(int(height/frame), 1, -1):
        mx = np.max(hist[x*frame-frame:x*frame])
        new_mx = np.max(hist[(x-1)*frame-frame:(x-1)*frame])
        indx2 = np.argmax(hist[x*frame-frame:x*frame]) + x * frame-frame

        if mx > new_mx and ((mx > border and indx2<indx) or (mx > first_max + 10 and indx2>indx)):

            indx2 = np.argmax(hist[x*frame-frame:x*frame]) + x*frame-frame + 3

            for i, kd in zip(listOfPossibleChars, range(len(listOfPossibleChars))):

                # height, width = imgThresh.shape
                # imgContoursd = np.zeros((height, width, 3), np.uint8)
                # lst = []
                # for j in listOfPossibleChars:
                #     lst.append(j.contour)
                # cv2.drawContours(imgContoursd, lst, k, Main.SCALAR_WHITE)
                # imgContoursd[:, indx2 + shift] = 255
                #
                # cv2.imshow('konturh', imgContoursd)
                #
                # cv2.waitKey(0)

                if indx2 + shift > i.intCenterX and indx2 +shift < i. intBoundingRectX + i.intBoundingRectWidth:
                    indx2 = i.intBoundingRectX + i.intBoundingRectWidth + 2 - shift
                    break
            if indx2 > height - 1:
                indx2 = height - 1
            imgThresh = imgThresh[:, :indx2]
            imgGrayScale = imgGrayScale[:, :indx2]

            break
    # cv2.imshow('loljjjjjjjjjjjdjj', imgThresh)
    # cv2.waitKey(0)

    # plt.plot(hist)
    # plt.plot([0, height], [border, border])
    # plt.plot([indx, indx], [np.max(hist), np.min(hist)])
    # #
    # #
    # plt.show()

    return imgThresh, imgGrayScale



MIN_PIXEL_WIDTH_PP = 3 ########################3
MIN_PIXEL_HEIGHT_PP = 15 ######################8

MIN_ASPECT_RATIO_PP = 0.1######################0.15
MAX_ASPECT_RATIO_PP = 1.5#######################1

MIN_PIXEL_AREA_PP = 70##########################100

def checkIfPossibleChar_PreProc(possibleChar):
    # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
    # note that we are not (yet) comparing the char to other chars to look for a group
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA_PP and
            possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH_PP and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT_PP and
            MIN_ASPECT_RATIO_PP < possibleChar.fltAspectRatio < MAX_ASPECT_RATIO_PP):
        return True
    else:
        return False


MIN_PIXEL_WIDTH_Br = 4 ########################3
MIN_PIXEL_HEIGHT_Br = 15 ######################8

MIN_ASPECT_RATIO_Br = 0.15######################0.15
MAX_ASPECT_RATIO_Br = 1

MIN_PIXEL_AREA_Br = 100

def checkIfPossibleChar_Br(possibleChar):
    # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
    # note that we are not (yet) comparing the char to other chars to look for a group
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA_Br and
            possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH_Br and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT_Br and
            MIN_ASPECT_RATIO_Br < possibleChar.fltAspectRatio < MAX_ASPECT_RATIO_Br):
        return True
    else:
        return False


def rotate(imgOriginal, listOfMatchingChars):
    listOfMatchingChars.sort(
        key=lambda matchingChar: matchingChar.intCenterX)  # sort chars from left to right based on x position

    # calculate the center point of the plate
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[
        len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[
        len(listOfMatchingChars) - 1].intCenterY) / 2.0
    # This is the probable centeral point of this plate.
    ptPlateCenter = fltPlateCenterX, fltPlateCenterY




    # calculate correction angle of plate region
    fltOpposite = listOfMatchingChars[-1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[-1])
    try:
        fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    except:
        fltCorrectionAngleInRad = 0
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)


    # final steps are to perform the actual rotation

    # get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg,
                                             1.0)  # The first poin tis the point of rotaion or center,theta and scaling factor

    height, width = imgOriginal.shape  # unpack original image width and height

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))  # rotate the entire image

    return imgRotated




def preprocessS_without_crop(imgOriginal):


    imgBl = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    imgGrayscale = copy.deepcopy(imgBl)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))

    im = clahe.apply(imgBl)
    # im = cv2.GaussianBlur(im, (3,3), 0)


    # im= cv2.fastNlMeansDenoising(imgBl, h=1, templateWindowSize=3, searchWindowSize=3)
    # cv2.imshow('plate', im)
    # cv2.waitKey(0)


    imgThresh = half_thresh(im)
    imgThresh = kray_fill_new(imgThresh)

    listOfPossibleChars, numberOfChars, noneChar = find_psbl_chr(imgThresh)
    if numberOfChars > 2:
        imgThresh = separate_stick(imgThresh, listOfPossibleChars, noneChar)
    listOfPossibleChars, _, _ = find_psbl_chr(imgThresh)



    # croped = False
    # if len(listOfPossibleChars) > 7 or len(listOfPossibleChars) == 0:
    croped = True
    #     # print('jhfewkhwefohvddcdd')
    # else:
    #     imgThresh = im.copy()
    imgThresh = half_thresh(im)
    imgThresh = kray_fill_I(imgThresh)


    imgForCount = imgThresh.copy()
    h, imgForCount = cv2.threshold(imgForCount, 1, 1, cv2.THRESH_BINARY_INV)

    listOfPossibleChars, _, _ = find_psbl_chr(imgThresh)

    # while True:
    if len(listOfPossibleChars) > 7:
        imgThresh = replace_last_first(imgThresh, listOfPossibleChars, len(listOfPossibleChars))

        imgThresh = del_blue(imgOriginal, listOfPossibleChars, imgThresh)


    listOfPossibleChars, _, _ = find_psbl_chr(imgThresh)


    imgThresh = draw_top_line(listOfPossibleChars, imgThresh)
    imgThresh = draw_btm_line(listOfPossibleChars, imgThresh)

    listOfPossibleChars, numberOfChars, noneChar = find_psbl_chr(imgThresh)

    if numberOfChars >= 2:
        imgThresh = separate_stick(imgThresh, listOfPossibleChars, noneChar)


    listOfPossibleChars, numberOfChars, noneChar = find_psbl_chr_4broken(imgThresh)

    imgThresh = matching_broken_chars(imgThresh, noneChar, imgForCount)



    # imgContours, contours, npaHierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # listOfPossibleChars = []
    #
    # # height, width = imgThresh.shape
    # # imgContours = np.zeros((height, width, 3), np.uint8)
    # for contour in range(len(contours)):                        # for each contour
    #     possibleChar = PossibleChar.PossibleChar(contours[contour])
    #     if DetectChars.checkIfPossibleChar(possibleChar):              # if contour is a possible char, note this does not compare to other chars (yet) . . .
    #         listOfPossibleChars.append(possibleChar)



    listOfPossibleChars, _, _ = find_psbl_chr(imgThresh)







    listOfPossibleChars, _, _ = find_psbl_chr(imgThresh)
    height, width = imgThresh.shape
    imgContours = np.zeros((height, width), np.uint8)
    imgForRcgnz = np.zeros((height, width), np.uint8)

    imgForRcgnzCopy = imgGrayscale.copy()
    for i in listOfPossibleChars:
        x1 = np.max([i.intBoundingRectX - 2, 0])
        y1 = np.min([i.intBoundingRectY + i.intBoundingRectHeight + 1, height-1])
        roi = imgForRcgnzCopy[:, x1: i.intBoundingRectX + i.intBoundingRectWidth + 2]

        h, imgForRcgnz[i.intBoundingRectY - 1:y1, x1: i.intBoundingRectX + i.intBoundingRectWidth + 2] = \
            cv2.threshold(roi[i.intBoundingRectY - 1:y1, :], 2, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h, imgContours[i.intBoundingRectY - 1:y1,
           x1: i.intBoundingRectX + i.intBoundingRectWidth + 2] = \
            cv2.threshold(roi[i.intBoundingRectY - 1:y1, :], 2, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)




    listOfPossibleChars, numberOfChars, noneChar = find_psbl_chr_4broken(imgContours)


    if numberOfChars >= 2:
        imgContours = separate_stick(imgContours, listOfPossibleChars, noneChar)

    listOfPossibleChars, numberOfChars, noneChar = find_psbl_chr_4broken(imgContours)
    imgContours = matching_broken_chars(imgContours, noneChar, imgForCount)

    kernel = np.ones((3, 3), np.uint8)
    imgForRcgnz = cv2.erode(imgForRcgnz, kernel)
    # imgContours =
    # cv2.imshow('Erod', imgContours)
    # cv2.waitKey(0)
    return imgGrayscale, imgContours, imgForRcgnz









MIN_PIXEL_WIDTH_I = 10 ########################3
MIN_PIXEL_HEIGHT_I = 20 ######################8

MIN_ASPECT_RATIO_I = 0.1######################0.15
MAX_ASPECT_RATIO_I = 1.5#######################1

MIN_PIXEL_AREA_I = 100

def checkIfPossibleChar_I(possibleChar):
    # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
    # note that we are not (yet) comparing the char to other chars to look for a group
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA_I and
            possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH_I and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT_I and
            MIN_ASPECT_RATIO_I < possibleChar.fltAspectRatio < MAX_ASPECT_RATIO_I):
        return True
    else:
        return False

def find_psbl_chr_intrnl(imgThresh):
    listOfPossibleChars = []
    noneChar = []
    imgThreshCopy = imgThresh.copy()

            # find all contours in plate
    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    Ar = imgThresh.shape[0] * imgThresh.shape[1]
    for contour in range(len(contours)):                        # for each contour
        possibleChar = PossibleChar.PossibleChar(contours[contour])
        min_ar = possibleChar.intBoundingRectHeight * possibleChar.intBoundingRectWidth
        print(min_ar/Ar)
        if checkIfPossibleChar_I(possibleChar) and 0.04 < min_ar/Ar < 0.1 :              # if contour is a possible char, note this does not compare to other chars (yet) . . .
            listOfPossibleChars.append(possibleChar)
        else:
            noneChar.append(possibleChar)
    numberOfChars = len(listOfPossibleChars)
    listOfPossibleChars = sorted(listOfPossibleChars, key=lambda obj: obj.intBoundingRectArea)
    return listOfPossibleChars, numberOfChars, noneChar


def kray_fill_I(imgThresh):
    imgForCount = imgThresh.copy()
    h, imgForCount = cv2.threshold(imgForCount, 1, 1, cv2.THRESH_BINARY_INV)

    ln = imgThresh.shape[1]
    counterU = 0
    counterB = imgThresh.shape[0] - 1
    counterL = 0
    counterR = imgThresh.shape[1] - 1

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
    if counterR < imgThresh.shape[1] / 100 * 80:
        counterR = 0
    elif counterL > imgThresh.shape[1] / 100 * 20:
        counterL = imgThresh.shape[1] - 1
    elif counterU > imgThresh.shape[0] / 100 * 20:
        counterU = imgThresh.shape[0] - 1
    elif counterB < imgThresh.shape[0] / 100 * 80:
        counterB = 0

    imgThresh[:counterU] = 0
    imgThresh[counterB:] = 0
    imgThresh[:, :counterL] = 0
    imgThresh[:, counterR:] = 0

    return imgThresh