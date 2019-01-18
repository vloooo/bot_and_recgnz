import cv2
import numpy as np
import math
import Preprocess
import PossibleChar
import record_webhook


def detect_chars_in_plates(list_of_possible_plates):
    if len(list_of_possible_plates) == 0:  # if list of possible plates is empty
        return list_of_possible_plates

    list_of_possible_plates_refined = []
    # handle both photo to finding characters, and recognize chr
    for psbl_plate in list_of_possible_plates:
        # strict preprocessing
        psbl_plate.img_gray, psbl_plate.img_thresh, psbl_plate.img_4_rcgnz = \
            Preprocess.preprocess_for_plate(psbl_plate.img_plate)

        # resizing
        psbl_plate.img_thresh = cv2.resize(psbl_plate.img_thresh, (0, 0), fx=1.6, fy=1.6,
                                           interpolation=cv2.INTER_LINEAR)
        psbl_plate.img_4_rcgnz = cv2.resize(psbl_plate.img_4_rcgnz, (0, 0), fx=1.6, fy=1.6,
                                            interpolation=cv2.INTER_LINEAR)

        # thresholding
        _, psbl_plate.img_thresh = cv2.threshold(psbl_plate.img_thresh, 0.0, 255.0,
                                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, psbl_plate.img_4_rcgnz = cv2.threshold(psbl_plate.img_4_rcgnz, 0.0, 255.0,
                                                  cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # find all possible chars in the plate,
        possible_chars_in_plate, _, _ = find_psbl_chr(psbl_plate.img_thresh, [70, 3, 15, 0.1, 1.5])

        # find groups of matching chars within the plate
        matching_chars_in_plate = \
            find_all_cmbn_mtch_chrs(possible_chars_in_plate, psbl_plate.img_thresh, [10., 33., 1, 1, 0.3], False, False)

        # if didn't find groups of matching char pass empty string
        if len(matching_chars_in_plate) == 0:
            psbl_plate.strChars = ""
            continue

        # remove_overlapping_chars
        for i in range(0, len(matching_chars_in_plate)):
            matching_chars_in_plate[i].sort(key=lambda matching_char: matching_char.center_x)
            matching_chars_in_plate[i] = remove_overlapping_chars(matching_chars_in_plate[i])

        longest_plate = find_longest(matching_chars_in_plate)

        psbl_plate.strChars = recognize_chars(psbl_plate.img_4_rcgnz, longest_plate)

        list_of_possible_plates_refined.append(psbl_plate)

    return list_of_possible_plates_refined


###################################################################################################
def find_longest(matching_chars_in_plate):
    # within each possible plate, suppose the longest list of potential matching chars is the actual list of chars
    longest_plate = 0
    true_plate_index = 0

    # loop through all the vectors of matching chars, get the index of the one with the most chars
    for i in range(0, len(matching_chars_in_plate)):
        if len(matching_chars_in_plate[i]) > longest_plate:
            longest_plate = len(matching_chars_in_plate[i])
            true_plate_index = i
    longest_plate = matching_chars_in_plate[true_plate_index]

    return longest_plate


def find_all_cmbn_mtch_chrs(possible_chars, img, params, extra_cndtion=True, sorting=True):
    """
    find several sequences of countours, that can be plate
    """

    # sorting and filtering contours
    all_mtch_chars = []
    uniq_chars = list(set(possible_chars))
    possible_chars = sorted(uniq_chars, key=lambda x: x.center_x)

    # find char seq for firs char
    for psb_ch in possible_chars:
        matched_chrs = find_chr_seq(psb_ch, possible_chars, img, params, extra_cndtion, sorting)
        matched_chrs.append(psb_ch)

        # if current possible list of matching chars is not long enough to constitute a possible plate
        if len(matched_chrs) < 4:
            continue
        # remember found seq and won't analise its characters again
        all_mtch_chars.append(matched_chrs)
        none_matched_chrs = list(set(possible_chars) - set(matched_chrs))

        # find another sequences recursively
        extra_mtching_chars = find_all_cmbn_mtch_chrs(none_matched_chrs, img, params)
        for i in extra_mtching_chars:
            all_mtch_chars.append(i)
        break

    return all_mtch_chars


def find_chr_seq(psb_ch, list_of_chars, img, params, extra_cndtion=True, sorting=True):
    """
    the purpose of this function is, given a possible char and a big list of possible chars,
    find all chars in the big list that are a match for the single possible char, and return those matching chars
    """

    matched_chrs = []

    center_x = [psb_ch.center_x]
    tops = [psb_ch.pos_y]
    btms = [psb_ch.pos_y + psb_ch.height]

    # sort chars from left to right
    if sorting:
        matched_chrs = sorted(matched_chrs, key=lambda x: x.center_x)

    # find similar chrs
    for pretending_ch in list_of_chars:
        if pretending_ch == psb_ch or pretending_ch in matched_chrs:
            continue

        # find differences between chrs
        change_in_distance = distance_between_chars(psb_ch, pretending_ch)
        change_in_angel = angle_between_chars(psb_ch, pretending_ch)
        change_in_area = float(abs(pretending_ch.area - psb_ch.area)) / float(psb_ch.area)
        change_in_width = float(abs(pretending_ch.width - psb_ch.width)) / float(psb_ch.width)
        change_in_height = float(abs(pretending_ch.height - psb_ch.height)) / float(psb_ch.height)

        # find differences in position on image
        extra_key = True
        if extra_cndtion:
            extra_key = abs(pretending_ch.pos_x - np.mean(center_x)) < img.shape[1] / 100 * 20 and \
                        abs(pretending_ch.pos_y - np.mean(tops)) < img.shape[0] / 100 * 3 and \
                        abs((pretending_ch.pos_y + pretending_ch.height) - np.mean(btms)) < img.shape[0] / 100 * 2.5

        # if differs is insensible, match two chrs
        if (change_in_distance < (psb_ch.diagonal_size * params[0]) and
                change_in_angel < params[1] and change_in_area < params[2] and
                change_in_width < params[3] and change_in_height < params[4] and extra_key):

            # matching chars and update approximate plate position on image
            center_x.append(pretending_ch.center_x)
            tops.append(pretending_ch.pos_y)
            btms.append(pretending_ch.pos_y + pretending_ch.height)
            matched_chrs.append(pretending_ch)

    return matched_chrs


# use Pythagorean theorem to calculate distance between two chars
def distance_between_chars(first_char, second_char):
    dif_x = abs(first_char.center_x - second_char.center_x)
    dif_y = abs(first_char.center_y - second_char.center_y)

    return math.sqrt((dif_x ** 2) + (dif_y ** 2))


# use basic trigonometry (SOH CAH TOA) to calculate angle between chars
def angle_between_chars(first_char, second_char):
    adj = float(abs(first_char.center_x - second_char.center_x))
    flt_opp = float(abs(first_char.center_y - second_char.center_y))

    if adj != 0.0:
        flt_angle_in_rad = math.atan(flt_opp / adj)  # if adjacent is not zero, calculate angle
    else:
        flt_angle_in_rad = 1.5708

    flt_angle_in_deg = flt_angle_in_rad * (180.0 / math.pi)  # calculate angle in degrees

    return flt_angle_in_deg


def remove_overlapping_chars(matched_chrs):
    """
# if we have two chars overlapping or to close to each other to possibly be separate chars, remove the inner char,
# this is to prevent including the same char twice if two contours are found for the same char,
# for example for the letter 'O' both the inner ring and the outer ring may be found as contours
    """
    matched_chrs_to_remove = list(matched_chrs)  # this will be the return value

    for current_char in matched_chrs:
        for other_char in matched_chrs:
            if current_char != other_char:
                if (distance_between_chars(current_char, other_char) <
                        current_char.diagonal_size * 0.2):

                    if current_char.area < other_char.area:  # if current char is smaller than other char
                        if current_char in matched_chrs_to_remove:
                            matched_chrs_to_remove.remove(current_char)
                    else:
                        if other_char in matched_chrs_to_remove:
                            matched_chrs_to_remove.remove(other_char)

    return matched_chrs_to_remove


def recognize_chars(img_thresh, matched_chrs):
    """
    recognize characters by neural network and manual tests
    """
    # preparing photo to recognizing
    height, width = img_thresh.shape
    img_thresh_color = np.zeros((height, width, 3), np.uint8)
    _, img_thresh = cv2.threshold(img_thresh, 0.0, 255.0, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR, img_thresh_color)

    # sort chars from left to right
    matched_chrs.sort(key=lambda x: x.center_x)

    # find average width of chrs
    str_chars = ""
    sm = 0
    for i in matched_chrs:
        sm += i.width
    mean = sm / len(matched_chrs)

    for current_char in matched_chrs:

        # crop letter
        roi = img_thresh_color[current_char.pos_y: current_char.pos_y + current_char.height,
              current_char.pos_x: current_char.pos_x + current_char.width]
        roi_gray = img_thresh[current_char.pos_y: current_char.pos_y + current_char.height,
                   current_char.pos_x: current_char.pos_x + current_char.width]

        roi = cv2.copyMakeBorder(roi, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        roi_gray = cv2.copyMakeBorder(roi_gray, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_LINEAR)

        # predict by network
        img = np.reshape(roi, [1, 64, 64, 3])
        classes = record_webhook.model.predict_classes(img)

        # manual analysis
        if classes[0] == 17 or classes[0] == 22 or classes[0] == 23 or classes[0] == 20 or classes[0] == 18 \
                or classes[0] == 31 or classes[0] == 33:

            img_for_count = crop_letter(roi_gray)
            top_ar, bot_ar, top_sum, bot_sum, cn = find_full_lines(img_for_count)

            # check for M
            m_key = True
            if len(top_ar) > 0 and top_ar[-1] / top_ar[0] < 0.45 and len(bot_ar) > 0 and (
                    bot_ar[0] / bot_ar[-1] < 0.85 or bot_ar[0] / bot_ar[-1] > 1) and \
                    cn > 1 and (classes[0] == 17 or classes[0] == 20 or classes[0] == 22):
                classes[0] = 22
                m_key = False

            # check for W
            if top_sum > bot_sum and 0.7 > bot_sum / top_sum > 0.1 and m_key:
                classes[0] = 32

        # check for 1
        if current_char.width < mean - 10:
            classes[0] = 18

        # check for A || L
        if classes[0] == 21 or classes[0] == 10:
            img_for_count = crop_letter(roi_gray)
            top_ar, bot_ar, top_sum, bot_sum, _ = find_full_lines(img_for_count)
            if len(top_ar) > 0 and len(bot_ar) > 0 and bot_sum > 1 and top_sum > 1:
                classes[0] = 10
            else:
                classes[0] = 21

        # check for O || U
        if classes[0] == 24 or classes[0] == 30 or classes[0] == 0:
            img_for_count = crop_o_u(roi_gray)
            area = img_for_count.shape[0] * img_for_count.shape[1]
            top_ar, bot_ar, top_sum, bot_sum, _ = find_full_lines(img_for_count, prcnt=65)

            if len(top_ar) > 0 and top_sum / area > .1:
                classes[0] = 30
            else:
                classes[0] = 24

        # get character from results
        if classes[0] < 10:
            str_current_char = chr(classes[0] + 48)
        else:
            str_current_char = chr(classes[0] + 55)

        str_chars = str_chars + str_current_char
    return str_chars


def crop_letter(roi):
    """
    crop letter from roi rightly by contour
    """
    # prepare image for count black pixels
    img_for_count = roi.copy()
    _, img_for_count = cv2.threshold(img_for_count, 1, 1, cv2.THRESH_BINARY_INV)

    # find sum by lines and colls
    sm_lines = img_for_count.sum(axis=1)
    sm_cols = img_for_count.sum(axis=0)

    top = 0
    bottom = img_for_count.shape[0] - 1
    left = 0
    right = img_for_count.shape[1] - 1

    # find not empty lines and columns
    for i in range(len(sm_lines) - 1):
        if sm_lines[i] != 0:
            top = i
            break

    for i in range(len(sm_lines) - 1, 0, -1):
        if sm_lines[i] != 0:
            bottom = i
            break

    for i in range(len(sm_cols) - 1):
        if sm_cols[i] != 0:
            left = i
            break

    for i in range(len(sm_cols) - 1, 0, -1):
        if sm_cols[i] != 0:
            right = i
            break

    # crop letter
    img_for_count = img_for_count[top:bottom, left:right]

    return img_for_count


def crop_o_u(roi):
    """
    like crop_letter, but find columns filling for 40%
    """
    img_for_count = roi.copy()
    _, img_for_count = cv2.threshold(img_for_count, 1, 1, cv2.THRESH_BINARY_INV)
    sm_lines = img_for_count.sum(axis=1)
    sm_cols = img_for_count.sum(axis=0)

    top = 0
    bottom = img_for_count.shape[0] - 1
    left = 0
    right = img_for_count.shape[1] - 1

    for i in range(len(sm_lines) - 1):
        if sm_lines[i] != 0:
            top = i
            break

    for i in range(len(sm_lines) - 1, 0, -1):
        if sm_lines[i] != 0:
            bottom = i
            break

    for i in range(len(sm_cols) - 1):
        if sm_cols[i] / img_for_count.shape[0] > .4:
            left = i
            break

    for i in range(len(sm_cols) - 1, 0, -1):
        if sm_cols[i] / img_for_count.shape[0] > .4:
            right = i
            break

    img_for_count = img_for_count[top:bottom, left:right]
    return img_for_count


def find_full_lines(img_for_count, prcnt=80):
    """
    find number of white pixels upper and lower filled for some% --> full lines
    """
    sm_lines = img_for_count.sum(axis=1)

    im_len = img_for_count.shape[1]
    maximums = np.where(sm_lines > im_len / 100 * prcnt, 1, 0)
    top_sum = 1
    bot_sum = 1
    top_ar = []
    bot_ar = []
    switch_key = False

    # filing top array and increase top_sum before full line was founded, after filling bottom array/bot_sum
    for line, key in zip(img_for_count, maximums):

        tmp = 1
        cur_st = False
        start_key = False
        if not key:

            for i in range(len(line) - 2):
                if line[i] == 1 and line[i + 1] == 0:
                    start_key = True

                elif line[i] == 0 and line[i + 1] == 1 and start_key:
                    start_key = False

                    if switch_key:
                        bot_sum += tmp
                        if not cur_st:
                            bot_ar.append(tmp)
                        else:
                            bot_ar[-1] += tmp
                    else:
                        top_sum += tmp
                        if not cur_st:
                            top_ar.append(tmp)
                        else:
                            top_ar[-1] += tmp

                    cur_st = True
                    tmp = 1

                if start_key:
                    tmp += 1
        else:
            switch_key = True

################################################################################
    # find how many times image changes from white to black, after full lines
    countr = find_m_tail(img_for_count, maximums)

    return top_ar, bot_ar, top_sum, bot_sum, countr


def find_m_tail(img_for_count, maximums):
    """
    find how many times image changes from white to black, after full lines
    """
    maximums = list(maximums)
    try:
        countr = []
        indx = len(maximums) - 1 - maximums[::-1].index(1) + 3

        for j in img_for_count[indx:]:
            start_key = False
            line = j
            countr.append(0)
            for i in range(len(line) - 2):
                if line[i] == 1 and line[i + 1] == 0:
                    start_key = True
                elif line[i] == 0 and line[i + 1] == 1 and start_key:
                    start_key = False
                    countr[len(countr) - 1] += 1
        countr = np.max(countr)
    except:
        countr = 0

    return countr


def find_psbl_chr(img_thresh, psbl_char_params, explore_area=False):
    """
    check all contours on image if can they be chrs
    """
    list_of_possible_chars = []
    none_char = []

    # find all contours in plate
    _, contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plate_area = img_thresh.shape[0] * img_thresh.shape[1]

    for contour in range(len(contours)):
        # make simple contour like instance of PossibleChar
        pssbl_chr = PossibleChar.PossibleChar(contours[contour])
        char_area = pssbl_chr.height * pssbl_chr.width

        # checking for: can contour be a character
        if not explore_area and check_if_pssbl_chr(pssbl_chr, psbl_char_params):
            list_of_possible_chars.append(pssbl_chr)
        elif explore_area and 0.04 < char_area / plate_area < 0.1 and check_if_pssbl_chr(pssbl_chr, psbl_char_params):
            list_of_possible_chars.append(pssbl_chr)
        else:
            none_char.append(pssbl_chr)

    number_of_chars = len(list_of_possible_chars)
    list_of_possible_chars = list(sorted(list_of_possible_chars, key=lambda obj: obj.area))

    return list_of_possible_chars, number_of_chars, none_char


def check_if_pssbl_chr(psb_char, params):
    """
    checking for: can contour be a character
    """
    if (psb_char.area > params[0] and psb_char.width > params[1] and psb_char.height > params[2] and
            params[3] < psb_char.aspect_ratio < params[4]):
        return True
    else:
        return False

# _P[120, 8, 10, 0.25, 1.5]
