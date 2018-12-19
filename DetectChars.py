# DetectChars.py

import cv2
import numpy as np
import math
import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
import Main
from PIL import Image
import Preprocess
import PossibleChar
import webhooks
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.optimizers import RMSprop


np.set_printoptions(threshold=np.nan)

# module level variables ##########################################################################
        # constants for checkIfPossibleChar, this checks one possible char only (does not compare to another char)
MIN_PIXEL_WIDTH = 3
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.1
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 100

        # constants for comparing two chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.2
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.5 ###########0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

        # other constants
MIN_NUMBER_OF_MATCHING_CHARS = 4

RESIZED_CHAR_IMAGE_WIDTH = 64
RESIZED_CHAR_IMAGE_HEIGHT = 64

MIN_CONTOUR_AREA = 100
model = load_model('char-reg.h5')
###################################################################################################
def loadCNNClassifier():
    model.compile(optimizer = RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.005), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return True###################################################################################################
def detectCharsInPlates(listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossiblePlates) == 0:          # if list of possible plates is empty
        return listOfPossiblePlates             # return
    # end if

            # at this point we can be sure the list of possible plates has at least one plate
    listOfPossiblePlates_refined = []
    for possiblePlate in listOfPossiblePlates:          # for each possible plate, this is a big for loop that takes up most of the function
        #possiblePlate.imgPlate = cv2.fastNlMeansDenoisingColored(possiblePlate.imgPlate,None,15,15,7,21)
        # #possiblePlate.imgPlate = cv2.equalizeHist(possiblePlate.imgPlate)
        possiblePlate.imgGrayscale, possiblePlate.imgThresh, possiblePlate.imgThreshForRcgnz = Preprocess.preprocessS(possiblePlate.imgPlate)     # preprocess to get grayscale and threshold images
        # cv2.imshow('laps', possiblePlate.imgThresh )
        # cv2.waitKey(0)

        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6,interpolation=cv2.INTER_LINEAR)
        possiblePlate.imgThreshForRcgnz = cv2.resize(possiblePlate.imgThreshForRcgnz, (0, 0), fx=1.6, fy=1.6, interpolation=cv2.INTER_LINEAR)

        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresholdValue, possiblePlate.imgThreshForRcgnz = cv2.threshold(possiblePlate.imgThreshForRcgnz, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


        '''
        # This clears the image more removing all the unknown noise from it.
        if Main.showSteps == True: # show steps ###################################################
            Image.fromarray(possiblePlate.imgThresh).show()
            input('Press Enter to Continue....')
        # end if # show steps #####################################################################
        '''
                # find all possible chars in the plate,
                # this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)


        '''
        if  Main.showSteps == True: # show steps ###################################################
            height, width, numChannels = possiblePlate.imgPlate.shape
            imgContours = np.zeros((500, 500, 3), np.uint8)
            del contours[:]                                         # clear the contours list

            for possibleChar in listOfPossibleCharsInPlate:
                contours.append(possibleChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
            #print('These are the possible characters in the plate :')
            cv2.imshow('lll', imgContours)
            cv2.waitKey(0)
        # end if # show steps #####################################################################
        '''

                # given a list of all possible chars, find groups of matching chars within the plate
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingCharsC(listOfPossibleCharsInPlate)
        if (len(listOfListsOfMatchingCharsInPlate) == 0):            # if no groups of matching chars were found in the plate
            '''
            if Main.showSteps == True: # show steps ###############################################
                print("chars found in plate number " + str(intPlateCounter) + " = (none), click on any image and press a key to continue . . .")
                intPlateCounter = intPlateCounter + 1
            '''

            possiblePlate.strChars = ""
            continue                        # go back to top of for loop
        # end if
        '''
        if Main.showSteps == True: # show steps ###################################################
            imgContours = np.zeros((300, 300, 3), np.uint8)
            del contours[:]

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for
                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            imgContours = Image.fromarray(imgContours,'RGB')
            imgContours.show()
            input('Press Enter to Continue....')
        # end if # show steps #####################################################################
        '''
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):                              # within each list of matching chars
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])              # and remove inner overlapping chars
        # end for
        '''
        if Main.showSteps == True: # show steps ###################################################
            imgContours = np.zeros((height, width, 3), np.uint8)

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                del contours[:]

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for

                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            imgContours = Image.fromarray(imgContours,'RGB')
            imgContours.show()
            input('Press Enter to Continue....')
        # end if # show steps #####################################################################
        '''

                # within each possible plate, suppose the longest list of potential matching chars is the actual list of chars
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

                # loop through all the vectors of matching chars, get the index of the one with the most chars
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
            # end if
        # end for

                # suppose that the longest list of matching chars within the plate is the actual list of chars
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        if Main.showSteps == True: # show steps ###################################################
            imgContours = np.zeros((300, 500, 3), np.uint8)
            del contours[:]

            for matchingChar in longestListOfMatchingCharsInPlate:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
            # imgContours = Image.fromarray(imgContours,'RGB')
            # imgContours.show()
            """
            cv2.imshow("The_Longest_list_of_matching_chars", imgContours)
            cv2.waitKey(0)
            """
        # end if # show steps #####################################################################

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThreshForRcgnz, longestListOfMatchingCharsInPlate)
        #print('this character is recognized :',possiblePlate.strChars)
        listOfPossiblePlates_refined.append(possiblePlate)

        if Main.showSteps == True: # show steps ###################################################
            print("chars found in plate number " + str(intPlateCounter) + " = " + possiblePlate.strChars + ", click on any image and press a key to continue . . .")
            intPlateCounter = intPlateCounter + 1
        # end if # show steps #####################################################################

    # end of big for loop that takes up most of the function

    if Main.showSteps == True:
        print("\nchar detection complete, click on any image and press a key to continue . . .\n")
    # end if
    return listOfPossiblePlates_refined # we return the list of plates with the probable plate number of each plate.

###################################################################################################
def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []                        # this will be the return value
    contours = []
    imgThreshCopy = imgThresh.copy()

            # find all contours in plate
    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for contour in range(len(contours)):                        # for each contour
        possibleChar = PossibleChar.PossibleChar(contours[contour])

        if checkIfPossibleChar(possibleChar):              # if contour is a possible char, note this does not compare to other chars (yet) . . .
            listOfPossibleChars.append(possibleChar)


            # cv2.drawContours(imgContours, contours, contour, Main.SCALAR_WHITE)
            # cv2.imshow('kontur', imgContours)
            # cv2.waitKey(0)
            # add to list of possible chars
        # end if
    # end if

    return listOfPossibleChars
# end function

###################################################################################################
def checkIfPossibleChar(possibleChar):
            # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
            # note that we are not (yet) comparing the char to other chars to look for a group
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False


###################################################################################################
def findListOfListsOfMatchingChars(listOfPossibleChars , img):
    listOfListsOfMatchingChars = []  # this will be the return value
    # print("Now we check which contours are similar")



    listOfChars = list(set(listOfPossibleChars))
    listOfPossibleChars = sorted(listOfChars, key=lambda x: x.intCenterX)

    for possibleChar in listOfPossibleChars:  # for each possible char in the one big list of chars

        # print('We are checking for :')
        # imgContours = np.zeros((height, width, 3), np.uint8)
        # cv2.drawContours(imgContours, possibleChar.contour, -1, Main.SCALAR_WHITE)
        # cv2.imshow("2b", imgContours)
        # cv2.waitKey(0)

        listOfMatchingChars = findListOfMatchingChars(possibleChar,
                                                      listOfPossibleChars, img)  # find all chars in the big list that match the current char
        listOfMatchingChars.append(possibleChar)  # also add the current char to current possible list of matching chars
        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:  # if current possible list of matching chars is not long enough to constitute a possible plate
            continue
        listOfListsOfMatchingChars.append(listOfMatchingChars)
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))
        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved, img)
        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:  # for each list of matching chars found by recursive call
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)
        break;

    return listOfListsOfMatchingChars

###################################################################################################
def findListOfMatchingChars(possibleChar, listOfChars, img):
            # the purpose of this function is, given a possible char and a big list of possible chars,
            # find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
    listOfMatchingChars = []                # this will be the return value

    # for k in listOfChars:
    #     print(k.intCenterX)
    # print('tttt')
    center = []
    center.append(possibleChar.intCenterX)
    # print('\n\n')
    # imgContours = np.zeros([img.shape[0], img.shape[1], 3])
    for possibleMatchingChar in listOfChars:                # for each char in big list
        if possibleMatchingChar == possibleChar or possibleMatchingChar in listOfMatchingChars:    # if the char we attempting to find matches for is the exact same char
                                                    # as the char in the big list we are currently checking
                                                    # then we should not include it in the list of matches b/c that would
                                                    # end up double including the current char
            continue                                # so do not add to list of matches and jump back to top of for loop
        # end if
                    # compute stuff to see if chars are a match
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

                # check if chars match
        # print(sum(center) / len(center), abs(possibleMatchingChar.intCenterX - sum(center) / len(center)),
        #       img.shape[1] / 100 * 20)

        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT) and\
            abs(possibleMatchingChar.intBoundingRectX - sum(center) / len(center)) < img.shape[1] / 100 *15:

            center.append(possibleMatchingChar.intCenterX)

            listOfMatchingChars.append(possibleMatchingChar)        # if the chars are a match, add the current char to list of matching chars
            # print("\t This contour is same:")
            # cv2.drawContours(imgContours, possibleMatchingChar.contour, -1, Main.SCALAR_WHITE)
            # cv2.imshow("2b", imgContours)
            # cv2.waitKey(0)
            # imgContours = np.zeros([img.shape[0], img.shape[1], 3])

        # end if
    # end for

    return listOfMatchingChars                  # return result
# end function

# use Pythagorean theorem to calculate distance between two chars
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))

# use basic trigonometry (SOH CAH TOA) to calculate angle between chars
def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:                           # check to make sure we do not divide by zero if the center X positions are equal, float division by zero will cause a crash in Python
        fltAngleInRad = math.atan(fltOpp / fltAdj)      # if adjacent is not zero, calculate angle
    else:
        fltAngleInRad = 1.5708                          # if adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this program
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       # calculate angle in degrees

    return fltAngleInDeg
# end function
###################################################################################################

# if we have two chars overlapping or to close to each other to possibly be separate chars, remove the inner (smaller) char,
# this is to prevent including the same char twice if two contours are found for the same char,
# for example for the letter 'O' both the inner ring and the outer ring may be found as contours, but we should only include the char once
def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)                # this will be the return value

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:        # if current char and other char are not the same char . . .
                                                                            # if current char and other char have center points at almost the same location . . .
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                                # if we get in here we have found overlapping chars
                                # next we identify which char is smaller, then if that char was not already removed on a previous pass, remove it
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:         # if current char is smaller than other char
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:              # if current char was not already removed on a previous pass . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)         # then remove current char
                        # end if
                    else:                                                                       # else if other char is smaller than current char
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:                # if other char was not already removed on a previous pass . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)           # then remove other char
                        # end if
                    # end if
                # end if
            # end if
        # end for
    # end for

    return listOfMatchingCharsWithInnerCharRemoved
# end function

###################################################################################################
# this is where we apply the actual char recognition
def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""               # this will be the return value, the chars in the lic plate

    height, width = imgThresh.shape
    imgThreshColor = np.zeros((height, width, 3), np.uint8)
    # imgThreshColor = imgThresh
    #imgThresh = cv2.cvtColor(imgThresh, cv2.COLOR_BGR2HSV)
    #imgHue, imgSaturation, imgThresh = cv2.split(imgHSV)
    #cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #imgThreshColor = imgThresh.copy()
    #imgThreshColor = cv2.resize(imgThreshColor, (0, 0), fx = 1.6, fy = 1.6)
    thresholdValue, imgThresh = cv2.threshold(imgThresh, 0.0, 255.0, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    imgThresh2 = imgThresh.copy()

    #imgThresh = cv2.fastNlMeansDenoising(imgThresh,None,10,10,7,21)
    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)
    #cv2.imshow('The Image',imgThreshColor)
    #cv2.waitKey(0)
    imgThreshColor2 = imgThreshColor.copy()
    #cv2.imshow('The Plate',imgThreshColor2)
    #cv2.waitKey(0)
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right
    sm = 0
    for i in listOfMatchingChars:
        sm += i.intBoundingRectWidth
    mean = sm / len(listOfMatchingChars)




    for currentChar in listOfMatchingChars:                                         # for each char in plate
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor2, pt1, pt2, (255,0,0), 2)           # draw green box around the char
                # crop char out of threshold image
        imgROI = imgThreshColor[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]
        imgROIgray = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        imgROI = cv2.copyMakeBorder(imgROI,8,8,8,8,cv2.BORDER_CONSTANT,value = [255,255,255])
        imgROIgray = cv2.copyMakeBorder(imgROIgray,8,8,8,8,cv2.BORDER_CONSTANT,value = [255,255,255])





        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)           # resize image, this is necessary for char recognition
        """
        cv2.imshow('letter', imgROI)
        cv2.waitKey(0)
        """


        """
        response = str(input('Want to save the image: '))
        if response == 'Y':
            name = str(input('Enter the name: '))
            cv2.imwrite(name, imgROIResized)
        """
        img=np.reshape(imgROIResized, [1, 64, 64, 3])
        classes=webhooks.model.predict_classes(img)

        if classes[0] == 17 or classes[0] == 22 or classes[0] == 23 or classes[0] == 20 or classes[0] == 18\
                or classes[0] == 31 or classes[0] == 33:
            imgForCount = crop_letter(imgROIgray)

            # print(botSum/topSum)
            # print(imgForCount)
            # print(topAr, '\n', botAr, '\n\n\n')
            topAr, botAr, topSum, botSum = find_full_lines(imgForCount)

            if len(topAr) > 0 and topAr[-1]/topAr[0] < 0.45 and len(botAr) > 0 and botAr[0]/botAr[-1] < 0.7 and classes[0] == 17:
                # print('MMMMMMMMMMMMMMMMMMMMM')
                classes[0] = 22

            if topSum > botSum and botSum/topSum < 0.77 and botSum/topSum > 0.1:
                classes[0] = 32

        if currentChar.intBoundingRectWidth < mean - 10:
            classes[0] = 18

        if classes[0] == 21 or classes[0] == 10:
            imgForCount = crop_letter(imgROIgray)
            topAr, botAr, topSum, botSum = find_full_lines(imgForCount)
            # print(topAr, botAr, topSum, botSum)
            if len(topAr)>0 and len(botAr) > 0 and botSum > 1 and topSum > 1:
                classes[0] = 10
            else:
                classes[0] = 21

        # print('\n\n\n', topSum, botSum)
        # if topSum > botSum:
        #     print('T', botSum/topSum*100)
        # else:
        #     if botSum != 0:
        #         print('B', topSum/botSum*100)


            # newRoi = imgROI[top:bottom, left:right]
            # cv2.imshow('llllllll', newRoi)
            # cv2.waitKey(0)



        if classes[0]<10:
            strCurrentChar = chr(classes[0]+48) # get character from results
        else:
            strCurrentChar = chr(classes[0]+55)    # get character from results
            # print(classes[0], strCurrentChar)

        if Main.showSteps == True:
            print(strCurrentChar, classes[0])
        strChars = strChars + strCurrentChar                        # append current char to full string


    # end for

    if Main.showSteps == True: # show steps #######################################################
        imgThreshColor2 = Image.fromarray(imgThreshColor2, 'RGB')
        imgThreshColor2.show()
        input('Press Enter to Continue....')
    # end if # show steps #########################################################################

    return strChars
# end function






def findListOfListsOfMatchingCharsC(listOfPossibleChars):
    listOfListsOfMatchingChars = []  # this will be the return value
    # print("Now we check which contours are similar")

    listOfPossibleChars = sorted(listOfPossibleChars, key=lambda obj: obj.intCenterX, reverse=True)


    for possibleChar in listOfPossibleChars:  # for each possible char in the one big list of chars

        # print('We are checking for :')
        # imgContours = np.zeros((height, width, 3), np.uint8)
        # cv2.drawContours(imgContours, possibleChar.contour, -1, Main.SCALAR_WHITE)
        # cv2.imshow("2b", imgContours)
        # cv2.waitKey(0)

        listOfMatchingChars = findListOfMatchingCharsC(possibleChar,
                                                      listOfPossibleChars)  # find all chars in the big list that match the current char
        listOfMatchingChars.append(possibleChar)  # also add the current char to current possible list of matching chars
        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:  # if current possible list of matching chars is not long enough to constitute a possible plate
            continue
        listOfListsOfMatchingChars.append(listOfMatchingChars)
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))
        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingCharsC(listOfPossibleCharsWithCurrentMatchesRemoved)
        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:  # for each list of matching chars found by recursive call
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)
        break;

    return listOfListsOfMatchingChars










MIN_DIAG_SIZE_MULTIPLE_AWAYC = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAYC = 7.0

MAX_CHANGE_IN_AREAC = 1

MAX_CHANGE_IN_WIDTHC = 0.8
MAX_CHANGE_IN_HEIGHTC = 0.3

MAX_ANGLE_BETWEEN_CHARSC = 21.0


def findListOfMatchingCharsC(possibleChar, listOfChars):


    listOfMatchingChars = []  # this will be the return value

    for possibleMatchingChar in listOfChars:  # for each char in big list
        if possibleMatchingChar == possibleChar:
            continue

        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)
        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(
            possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(
            abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(
            possibleChar.intBoundingRectWidth)

        fltChangeInHeight = float(
            abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(
            possibleChar.intBoundingRectHeight)

        # check if chars match
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAYC) and
                fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARSC and
                fltChangeInArea < MAX_CHANGE_IN_AREAC and
                fltChangeInWidth < MAX_CHANGE_IN_WIDTHC and
                fltChangeInHeight < MAX_CHANGE_IN_HEIGHTC):
            listOfMatchingChars.append(possibleMatchingChar)
            # print("\t This contour is same:")
            # cv2.drawContours(imgContours, possibleMatchingChar.contour, -1, Main.SCALAR_WHITE)
            # cv2.imshow("2b", imgContours)
            # cv2.waitKey(0)
        # end if
    # end for

    return listOfMatchingChars


def crop_letter(imgROIgray):
    imgForCount = imgROIgray.copy()
    cp = imgROIgray.copy()
    h, imgForCount = cv2.threshold(imgForCount, 1, 1, cv2.THRESH_BINARY_INV)
    smStr = imgForCount.sum(axis=1)
    smCol = imgForCount.sum(axis=0)

    top = 0
    bottom = imgForCount.shape[0] - 1
    left = 0
    right = imgForCount.shape[1] - 1

    for i in range(len(smStr) - 1):
        if smStr[i] != 0:
            # nonlocal top
            top = i
            break

    for i in range(len(smStr) - 1, 0, -1):
        if smStr[i] != 0:
            # nonlocal bottom
            bottom = i
            break

    for i in range(len(smCol) - 1):
        if smCol[i] != 0:
            # nonlocal left
            left = i
            break

    for i in range(len(smCol) - 1, 0, -1):
        if smCol[i] != 0:
            # nonlocal right
            right = i
            break

    # imgThresh = imgThresh[top:bottom, left:right]
    # cv2.imshow('yyy', cp[top:bottom, left:right])
    imgForCount = imgForCount[top:bottom, left:right]

    return imgForCount


def find_full_lines(imgForCount):
    smStr = imgForCount.sum(axis=1)
    # smCol = imgForCount.sum(axis=0)
    ln = imgForCount.shape[1]
    maximums = np.where(smStr > ln / 100 * 80, 1, 0)
    topSum = 1
    botSum = 1
    topAr = []
    botAr = []
    switchKey = False
    for str, key in zip(imgForCount, maximums):

        tmp = 1
        curSt = False
        startKey = False
        if not key:

            for i in range(len(str) - 2):
                if str[i] == 1 and str[i + 1] == 0:
                    startKey = True
                    # continue
                elif str[i] == 0 and str[i + 1] == 1 and startKey:
                    startKey = False

                    if switchKey:
                        botSum += tmp
                        if not curSt:
                            botAr.append(tmp)
                        else:
                            botAr[-1] += tmp
                        u = botAr[-1]
                    else:
                        topSum += tmp
                        if not curSt:
                            topAr.append(tmp)
                        else:
                            topAr[-1] += tmp
                        u = topAr[-1]
                    # print(str,'            ', u)

                    curSt = True
                    tmp = 1

                if startKey:
                    tmp += 1
        else:
            switchKey = True
    return topAr, botAr, topSum, botSum
