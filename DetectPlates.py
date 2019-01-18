import copy

import cv2
import math
import Preprocess
import DetectChars
import PossiblePlate
import PossibleChar

cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

###################################################################################################
def detect_plates_in_scene(img_originl):
    """
    detect plates
    """
    possible_plates = []

    img_gray, img_thresh_scene = Preprocess.preprocess_for_scene(img_originl)

    psb_chars = find_vld_chrs_by_cascad(img_thresh_scene, img_gray)

    # [6., 12., 1.2, 1.2, .3] alternative params
    all_matched_chars = DetectChars.find_all_cmbn_mtch_chrs(psb_chars, img_gray, [6., 12., 0.5, 1.2, .3])

    # for each group of matching chars, attempt to extract plate
    for matched_chars in all_matched_chars:
        psb_plate = extract_plate(img_originl, matched_chars)

        # if plate was found, add to list of possible plates
        if psb_plate.img_plate is not None:
            possible_plates.append(psb_plate)

    return possible_plates


def find_vld_chrs_by_cascad(img_thresh, gray):
    possible_chars = []

    # some bluring
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # find plates by standard Hoar's cascad
    plates = cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=3)

    # find contours on image
    _, contours, _ = cv2.findContours(img_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        #
        possible_char = PossibleChar.PossibleChar(contours[i])

        # if contour is a possible char, note this does not compare to other chars (yet) . . .
        if DetectChars.check_if_pssbl_chr(possible_char, [120, 8, 10, 0.25, 1.5]):
            for (x, y, w, h) in plates:
                if x < possible_char.pos_x < x + w and y < possible_char.center_y < y + h:
                    possible_chars.append(possible_char)  # and add to list of possible chars
                    break
    return possible_chars


def extract_plate(img_original, chars):
    possible_plate = PossiblePlate.PossiblePlate()  # this will be the return value

    chars.sort(key=lambda x: x.center_x)  # sort chars from left to right based on x position

    # calculate the center point of the plate
    plate_center_x = (chars[0].center_x + chars[-1].center_x) / 2.0
    plate_center_y = (chars[0].center_y + chars[-1].center_y) / 2.0

    # This is the probable centeral point of this plate.
    plate_center = plate_center_x, plate_center_y

    # calculate plate width and height
    plate_width = int((chars[-1].pos_x + chars[-1].width - chars[0].pos_x) * 1.3)
    # Here we calculate the probable width of this plate.
    height_sum = 0

    for matchingChar in chars:
        height_sum += matchingChar.height

    average_char_height = height_sum / len(chars)

    plate_height = int(average_char_height * 1.5)  # We include the padding factor.

    # calculate correction angle of plate region
    opposite = chars[-1].center_y - chars[0].center_y
    hypotenuse = DetectChars.distance_between_chars(chars[0], chars[-1])
    correction_angle_in_rad = math.asin(opposite / hypotenuse)
    correction_angle_deg = correction_angle_in_rad * (180.0 / math.pi)

    # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possible_plate.location_plate_on_img = (tuple(plate_center), (plate_width, plate_height), correction_angle_deg)

    # final steps are to perform the actual rotation
    # get the rotation matrix for our calculated correction angle
    rotation_matrix = cv2.getRotationMatrix2D(tuple(plate_center), correction_angle_deg, 1.0)

    height, width, _ = img_original.shape  # unpack original image width and height

    img_rotated = cv2.warpAffine(img_original, rotation_matrix, (width, height))  # rotate the entire image

    # imgTmp = cv2.getRectSubPix(img_rotated, (width, plate_height), tuple(plate_center))

    img_cropped = cv2.getRectSubPix(img_rotated, (plate_width, plate_height), tuple(plate_center))

    possible_plate.img_plate = img_cropped

    return possible_plate
