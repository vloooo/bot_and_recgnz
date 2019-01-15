import os
import urllib
import cv2
import numpy as np
import DetectChars
import DetectPlates
import Preprocess


def main(image_url):

    # # loading image
    resp = urllib.request.urlopen(image_url)
    image_url = np.asarray(bytearray(resp.read()), dtype="uint8")
    im_org = cv2.imdecode(image_url, cv2.IMREAD_COLOR)
    # im_org = cv2.imread(image_url)

    # croping useful area and resizing
    h, w = im_org.shape[:2]

    im_org = im_org[int(h / 100 *35): h - 20, 40: w - 40]
    im_org = cv2.resize(im_org, (0, 0), fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)

    if im_org is None:
        return 'AA00AAA'

    listOfPossiblePlates = DetectPlates.detect_plates_in_scene(im_org)  # detect plates

    if len(listOfPossiblePlates) > 1:
        listOfPossiblePlates = [choose_plate_to_handle(listOfPossiblePlates)]  # we don't need more than one plate

    listOfPossiblePlates = DetectChars.detect_chars_in_plates(listOfPossiblePlates)  # detect chars in plates

    # for i in listOfPossiblePlates:
    #     print(i.strChars, 'kkk')
    # plates with more than 8 symbols untruth
    listOfPossiblePlates = [x for x in listOfPossiblePlates if len(x.strChars) < 8]

    if len(listOfPossiblePlates) == 0:  # if no plates were found
        return 'AA00AAA'
    else:
        # plates with little number of symbols untruth
        listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)
        licPlate = listOfPossiblePlates[0]

        if len(licPlate.strChars) == 0:
            return 'AA00AAA'


    licPlate.strChars = validate_for_britain(licPlate.strChars)
    return licPlate.strChars


###################################################################################################
def validate_for_britain(line):
    """
    change network output to british plate's templates
    """
    ln = len(line)
    for j in range(ln):

        if line[j] == 'Q':
            line = line[:j] + "O" + line[j + 1:]

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
        elif line[j] == 'I' and j > 0 and j < ln - 4:
            line = line[:j] + "1" + line[j + 1:]
        elif line[j] == '1' and (j < 1 or j > ln - 4):
            line = line[:j] + "T" + line[j + 1:]
        elif line[j] == 'I' and (j < 1 or j > ln - 4):
            line = line[:j] + "W" + line[j + 1:]

        elif line[j] == 'Z' and ((ln == 7 and j > 1 and j < 4) or (ln == 6 and j > 0 and j < 3)):
            line = line[:j] + "2" + line[j + 1:]
        elif line[j] == '2' and (j < 1 or j > ln - 4):
            line = line[:j] + "Z" + line[j + 1:]

        elif line[j] == 'B' and ((ln == 7 and j > 1 and j < 4) or (ln == 6 and 0 < j < 3)):
            line = line[:j] + "8" + line[j + 1:]
        elif line[j] == '8' and (j < 1 or j > ln - 4):
            line = line[:j] + "B" + line[j + 1:]

        elif line[j] == '9' and (j < 1 or j > ln - 4):
            line = line[:j] + "B" + line[j + 1:]

        elif line[j] == '4' and (j < 1 or j > ln - 4):
            line = line[:j] + "A" + line[j + 1:]

    return line



def choose_plate_to_handle(list_of_psb_plates):
    """
    the idea of function is do some tests to find the most truthful plate
    """
    dif_list = find_most_likely_plts(list_of_psb_plates, 4, True, [100, 4, 15, 0.15, 1], False)

    new_list = [x for x in dif_list if x == 100]
    if len(new_list) == len(dif_list):
        dif_list = find_most_likely_plts(list_of_psb_plates, 5, False, [100, 10, 20, 0.1, 1.5], True)

    return list_of_psb_plates[np.argmin(dif_list)]


def find_most_likely_plts(list_of_psb_plates, number_chars_to_be_valid, explore_btm, psbl_char_params, explr_area):
    """
    the idea of function is do some tests to find the most truthful plate
    in falsePlates, preprocess_for_scene delete more than 20% of plate's image on one of edges
    """
    dif_list = []
    for image in list_of_psb_plates:

        image = image.img_plate
        image = Preprocess.preprocess_plate_without_croping(image)

        list_of_psb_chars, number_of_chars, _ = DetectChars.find_psbl_chr(image, psbl_char_params, explr_area)

        if number_of_chars < number_chars_to_be_valid:  # plates with small nmbr of symbols is untruth
            dif_list.append(1000)   # add some value bigger than psbl differ
            continue

        if explore_btm:
            dif_list = explore_btms(list_of_psb_chars, dif_list)
        else:
            dif_list.append(0)

    return dif_list


def explore_btms(list_of_psb_chars, dif_list):
    """
    bottom's differ of letters on plate's image usualy is less bottom's differ than falsePlate's image
    """
    list_of_psb_chars = sorted(list_of_psb_chars, key=lambda obj: obj.center_x)

    lttr_btms = []
    for i in list_of_psb_chars:
        lttr_btms.append(i.pos_y + i.height)
    mean_btm = np.mean(lttr_btms)

    df = []
    for i in list_of_psb_chars:
        df.append(abs(mean_btm - (i.pos_y + i.height)))

    dif_list.append(np.mean(df))

    return dif_list


if __name__ == "__main__":


    dirnrm = '/home/user/PycharmProjects/plates/cars/forTest'
    dirnrmM = '/home/user/PycharmProjects/plates/cars/new/norm'

    dirnrmN = '/home/user/PycharmProjects/plates/cars/new/neW' #  NL52SYZ)
    names = os.listdir(dirnrmN)
    counter = 0
    # names = ['WN08WDP.jpg']
    for i in names:
        c = main(dirnrmN + '/' + i)
        print(i[:-4],'         ', c)

        if i.find(c) != -1:
            print('!!!!!!!!!!!!')
            counter += 1

    print(counter, '/', len(names))
