# Main.py
import urllib

import cv2
import numpy as np
import os
import DetectChars
import DetectPlates

# module level variables ##########################################################################
import Preprocess

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False
counter = 0

def main(image):

    resp = urllib.request.urlopen(image)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    imgOriginalScene = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # imgOriginalScene = cv2.imread(image)  # open image

    h, w = imgOriginalScene.shape[:2]
    crop_img = imgOriginalScene[int(h / 100 *40): h - 40, 40: w - 40]


    imgOriginalScene = crop_img

    imgOriginalScene = cv2.resize(imgOriginalScene, (0, 0), fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)


    if imgOriginalScene is None:  # if image was not read successfully
        print("\nerror: image not read from file \n\n")  # print error message to std out
        # os.system("pause")  # pause so user can see error message
        return  # and exit program


    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)  # detect plates. We get a list of
    # combinations of contours that may be a plate.
    if len(listOfPossiblePlates) > 1:
        listOfPossiblePlates = [choose_plateE(listOfPossiblePlates)]

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)  # detect chars in plates

    listOfPossiblePlates = [x for x in listOfPossiblePlates if len(x.strChars)<8]



    if len(listOfPossiblePlates) == 0:  # if no plates were found
        print("\nno license plates were detected\n")  # inform user no plates were found
        response = 'AA00AAA'
        cv2.destroyAllWindows()
        return response, imgOriginalScene
    else:  # else
        # if we get in here list of possible plates has at leat one plate

        # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

        # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        licPlate = listOfPossiblePlates[0]


        if len(licPlate.strChars) == 0:  # if no chars were found in the plate
            print("\nno characters were detected\n\n")  # show message
            return 'AA00AAA', imgOriginalScene  # and exit program
        # end if


        licPlate.strChars = validate_for_britain(licPlate.strChars)
        cv2.destroyAllWindows()
        return licPlate.strChars, licPlate.imgPlate


###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(
        licPlate.rrLocationOfPlateInScene)  # get 4 vertices of rotated rect. Here, bounding rectangle is drawn with minimum area, so it considers the rotation also

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)  # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)


# end function

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0  # this will be the center of the area the text will be written to
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0  # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX  # choose a plain jane font
    fltFontScale = float(plateHeight) / 30.0  # base font scale on height of plate area
    intFontThickness = int(round(fltFontScale * 1.5))  # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale,
                                         intFontThickness)  # call getTextSize

    # unpack roatated rect into center point, width and height, and angle
    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),
     fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)  # make sure center is an integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)  # the horizontal location of the text area is the same as the plate

    if intPlateCenterY < (sceneHeight * 0.75):  # if the license plate is in the upper 3/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(
            round(plateHeight * 1.6))  # write the chars in below the plate
    else:  # else if the license plate is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(
            round(plateHeight * 1.6))  # write the chars in above the plate
    # end if

    textSizeWidth, textSizeHeight = textSize  # unpack text size width and height

    ptLowerLeftTextOriginX = int(
        ptCenterOfTextAreaX - (textSizeWidth / 2))  # calculate the lower left origin of the text area
    ptLowerLeftTextOriginY = int(
        ptCenterOfTextAreaY + (textSizeHeight / 2))  # based on the text area center, width, and height

    # write the text on the image
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace,
                fltFontScale, SCALAR_YELLOW, intFontThickness)


# end function

###################################################################################################
def validate_for_britain(line):
    ln = len(line)
    for j in range(len(line)):

        if line[j] == '0' and (j < 1 or j > ln - 4):
            line = line[:j] + "O" + line[j + 1:]
        elif line[j] == 'O' and ((ln == 7 and j > 0 and j < 4) or (ln == 6 and j > 0 and j < 3)):
            line = line[:j] + "0" + line[j + 1:]

        elif line[j] == 'S' and ((ln == 7 and j > 1 and j < 4) or (ln == 6 and j > 0 and j < 3)):
            line = line[:j] + "5" + line[j + 1:]
        elif line[j] == '5' and (j < 1 or j > ln - 4):
            line = line[:j] + "S" + line[j + 1:]

        elif (line[j] == 'T' or line[j] == 'Y' or line[j] == 'I' or line[j] == 'V') and (
                (ln == 7 and j > 1 and j < 4) or (ln == 6 and j > 0 and j < 3)):
            line = line[:j] + "1" + line[j + 1:]
        elif line[j] == '1' and (j < 1 or j > ln - 4):
            line = line[:j] + "T" + line[j + 1:]
        elif line[j] == 'I' and (j < 1 or j > ln - 4):
            line = line[:j] + "W" + line[j + 1:]

        elif line[j] == 'Z' and ((ln == 7 and j > 1 and j < 4) or (ln == 6 and j > 0 and j < 3)):
            line = line[:j] + "2" + line[j + 1:]
        elif line[j] == '2' and (j < 1 or j > ln - 4):
            line = line[:j] + "Z" + line[j + 1:]

        elif line[j] == 'B' and ((ln == 7 and j > 1 and j < 4) or (ln == 6 and j > 0 and j < 3)):
            line = line[:j] + "8" + line[j + 1:]
        elif line[j] == '8' and (j < 1 or j > ln - 4):
            line = line[:j] + "B" + line[j + 1:]

        elif line[j] == '9' and (j < 1 or j > ln - 4):
            line = line[:j] + "B" + line[j + 1:]

        elif line[j] == '4' and (j < 1 or j > ln - 4):
            line = line[:j] + "A" + line[j + 1:]
    return line



def choose_plateE(listOfPsbPlates):
    dif_list = []
    for image in listOfPsbPlates:

        image = image.imgPlate
        image_CP = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cp_CP = copy.deepcopy(image_CP)
        _, image, _ = Preprocess.preprocessS_without_crop(image)
        # cv2.imshow('image_masked.png', image)
        # cv2.waitKey(0)

        listOfPossibleChars, numberOfChars, noneChar = Preprocess.find_psbl_chr_4broken(image)
        # print(len(listOfPossibleChars))
        if len(listOfPossibleChars) < 4:
            dif_list.append(100)
            continue

        listOfPossibleChars = sorted(listOfPossibleChars, key=lambda obj: obj.intCenterX)

        let_area = []
        for i in listOfPossibleChars:
            let_area.append(i.intBoundingRectY + i.intBoundingRectHeight)
        Ar = np.mean(let_area)

        df = []
        for i in listOfPossibleChars:
            df.append(abs(Ar - (i.intBoundingRectY + i.intBoundingRectHeight)))
            # print(df)
        df = np.mean(df)
        dif_list.append(df)
    # print(dif_list)
    new_list = [x for x in dif_list if x == 100]
    if len(new_list) == len(dif_list):
        dif_list = []
        for image in listOfPsbPlates:

            image = image.imgPlate
            _, image, _ = Preprocess.preprocessS_without_crop(image)

            listOfPossibleChars, numberOfChars, noneChar = Preprocess.find_psbl_chr_intrnl(image)

            if len(listOfPossibleChars) < 5:
                dif_list.append(100)
                continue
            dif_list.append(0)

    return listOfPsbPlates[np.argmin(dif_list)]



if __name__ == "__main__":

    dirnrm = '/home/user/PycharmProjects/plates/cars/forTest'
    dirnrmN = '/home/user/PycharmProjects/plates/cars/new/norm'
    names = os.listdir(dirnrmN)
    counter = 0
    # names = ['home/user/PycharmProjects/plates/cars/forTest/A002BBV.jpg']
    for i in names:
        c, _ = main('/home/user/PycharmProjects/plates/cars/forTest/A002BBV.jpg')
        c = validate_for_britain(c)

        print(i[:-4], '         ', c)
        if i.find(c) != -1:
            print('!!!!!!!!!!!!')
            counter += 1
    print(counter)


