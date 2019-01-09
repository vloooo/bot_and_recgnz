import cv2
import numpy as np
import math
import DetectChars
import copy


def preprocess_for_scene(img_orig):
    output2 = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.adaptiveThreshold(output2, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9)

    return output2, img_thresh


def preprocess_for_plate(img_orig):
    img_bl = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    img_grayscale = copy.deepcopy(img_bl)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    im = clahe.apply(img_bl)

    list_of_psb_chr = prepare_to_hist_clcl(im)
    img_thresh, img_grayscale = histogram_kray_fill(im, img_grayscale, list_of_psb_chr)

    img_thresh = half_thresh(img_thresh)
    img_thresh = kray_fill(img_thresh)

    list_of_possible_chars, number_of_chars, _ = DetectChars.find_psbl_chr(img_thresh, [70, 3, 15, 0.1, 1.5])
    if number_of_chars > 7:
        img_thresh = replace_last_first(img_thresh, list_of_possible_chars, number_of_chars)
        img_thresh = del_blue(img_orig, list_of_psb_chr, img_thresh)

    list_of_psb_chr, _, _ = DetectChars.find_psbl_chr(img_thresh, [70, 3, 15, 0.1, 1.5])
    img_thresh = draw_top_line(list_of_psb_chr, img_thresh)
    img_thresh = draw_btm_line(list_of_psb_chr, img_thresh)

    list_of_psb_chr, nmbr_of_chrs, none_chars = DetectChars.find_psbl_chr(img_thresh, [70, 3, 15, 0.1, 1.5])
    if nmbr_of_chrs >= 2:
        img_thresh = separate_stick(img_thresh, list_of_psb_chr, none_chars)

    list_of_psb_chr, nmbr_of_chrs, none_chars = DetectChars.find_psbl_chr(img_thresh, [100, 4, 15, 0.15, 1])
    img_thresh = matching_broken_chars(img_thresh, none_chars)

    list_of_psb_chr, _, _ = DetectChars.find_psbl_chr(img_thresh, [70, 3, 15, 0.1, 1.5])
    height, width = img_thresh.shape
    img_thresh = np.zeros((height, width), np.uint8)
    img_for_rcgnz = np.zeros((height, width), np.uint8)

    img_for_rcgnz_copy = img_grayscale.copy()
    for i in list_of_psb_chr:
        x1 = np.max([i.pos_x - 2, 0])
        y1 = np.min([i.pos_y + i.height + 1, height - 1])

        roi = img_for_rcgnz_copy[:, x1: i.pos_x + i.width + 2]

        _, img_for_rcgnz[i.pos_y - 1:y1, x1: i.pos_x + i.width + 2] = \
            cv2.threshold(roi[i.pos_y - 1:y1, :], 2, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        _, img_thresh[i.pos_y - 1:y1, x1: i.pos_x + i.width + 2] = \
            cv2.threshold(roi[i.pos_y - 1:y1, :], 2, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    list_of_psb_chr, nmbr_of_chrs, none_chars = DetectChars.find_psbl_chr(img_thresh, [70, 3, 15, 0.1, 1.5])
    if nmbr_of_chrs > 0:
        img_thresh = rotate(img_thresh, list_of_psb_chr)
        img_for_rcgnz = rotate(img_for_rcgnz, list_of_psb_chr)

    _, img_thresh = cv2.threshold(img_thresh, 200, 255, cv2.THRESH_BINARY)
    list_of_psb_chr, nmbr_of_chrs, none_chars = DetectChars.find_psbl_chr(img_thresh, [70, 3, 15, 0.1, 1.5])
    img_thresh = draw_top_line(list_of_psb_chr, img_thresh)
    img_thresh = draw_btm_line(list_of_psb_chr, img_thresh)

    list_of_psb_chr, nmbr_of_chrs, none_chars = DetectChars.find_psbl_chr(img_thresh, [100, 4, 15, 0.15, 1])
    if nmbr_of_chrs >= 2:
        img_thresh = separate_stick(img_thresh, list_of_psb_chr, none_chars)

    list_of_psb_chr, nmbr_of_chrs, none_chars = DetectChars.find_psbl_chr(img_thresh, [100, 4, 15, 0.15, 1])
    img_thresh = matching_broken_chars(img_thresh, none_chars)

    kernel = np.ones((3, 3), np.uint8)
    img_for_rcgnz = cv2.erode(img_for_rcgnz, kernel)

    return img_grayscale, img_thresh, img_for_rcgnz


def prepare_to_hist_clcl(im):
    img_thresh = half_thresh(im)
    img_thresh = kray_fill(img_thresh, full_left_right=False)

    list_of_psb_chr, nmbr_of_chrs, none_chars = DetectChars.find_psbl_chr(img_thresh, [70, 3, 15, 0.1, 1.5])
    if nmbr_of_chrs > 2:
        img_thresh = separate_stick(img_thresh, list_of_psb_chr, none_chars)

    list_of_psb_chr, _, _ = DetectChars.find_psbl_chr(img_thresh, [70, 3, 15, 0.1, 1.5])
    return list_of_psb_chr


def half_thresh(img_grayscale):
    img_blurred = cv2.GaussianBlur(img_grayscale, (3, 3), 100)

    firs_half_img = img_blurred[:, :int(img_blurred.shape[1] / 2)]
    second_half_img = img_blurred[:, int(img_blurred.shape[1] / 2):]

    img_thresh = img_blurred.copy()
    _, img_thresh[:, :int(img_blurred.shape[1] / 2)] = cv2.threshold(firs_half_img, 0, 255,
                                                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, img_thresh[:, int(img_blurred.shape[1] / 2):] = cv2.threshold(second_half_img, 0, 255,
                                                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return img_thresh


def kray_fill(img_thresh, full_filling=False, full_left_right=True):
    img_for_count = img_thresh.copy()
    h, img_for_count = cv2.threshold(img_for_count, 1, 1, cv2.THRESH_BINARY_INV)

    ln = img_thresh.shape[1]
    counter_up = 0
    counter_bt = img_thresh.shape[0] - 1
    counter_lf = 1
    counter_rg = img_thresh.shape[1] - 1

    sm = img_for_count.sum(axis=1)
    for i in sm:
        counter_up += 1
        if i > int(ln / 100 * 40):
            break

    sm = sm[::-1]
    for i in sm:
        # p
        counter_bt -= 1
        if i > int(ln / 100 * 70):
            break

    ln = img_thresh.shape[0]
    sm = img_for_count.sum(axis=0)
    for i in sm:
        counter_lf += 1
        if i > int(ln / 100 * 60):
            break

    sm = sm[::-1]
    for i in sm:
        counter_rg -= 1
        if i > int(ln / 100 * 60):
            break
    if full_filling:
        if counter_rg < img_thresh.shape[1] / 100 * 80:
            counter_rg = 0
        elif counter_lf > img_thresh.shape[1] / 100 * 20:
            counter_lf = img_thresh.shape[1] - 1
        elif counter_up > img_thresh.shape[0] / 100 * 20:
            counter_up = img_thresh.shape[0] - 1
        elif counter_bt < img_thresh.shape[0] / 100 * 80:
            counter_bt = 0

    img_thresh[:counter_up] = 0
    img_thresh[counter_bt:] = 0
    if full_left_right:
        img_thresh[:, :counter_lf] = 0
        img_thresh[:, counter_rg:] = 0

    return img_thresh


def draw_top_line(list_of_psb_chr, img_thresh):
    if len(list_of_psb_chr):
        tops = [x.pos_y for x in list_of_psb_chr]
        btm_clear_line = np.max([int(np.mean(tops)), 0])
        img_thresh[btm_clear_line, :] = 0

    return img_thresh


def draw_btm_line(list_of_psb_chr, img_thresh):
    if len(list_of_psb_chr):
        btms = [x.pos_y + x.height for x in list_of_psb_chr]
        btm_clear_line = np.min([int(np.mean(btms)) + 1, img_thresh.shape[0] - 1])
        img_thresh[btm_clear_line, :] = 0

    return img_thresh


def matching_broken_chars(img_thresh, none_char):
    matched_none_char = []
    im_width = img_thresh.shape[1]
    im_height = img_thresh.shape[0]
    _, img_for_count = cv2.threshold(img_thresh, 1, 1, cv2.THRESH_BINARY_INV)

    for i in none_char:
        for j in none_char:
            if i != j and i not in matched_none_char and j not in matched_none_char:
                if abs(i.pos_x - j.pos_x) < im_width / 100 * 10 and abs(i.center_y - j.center_y) > im_height / 100 * 10\
                   and j.area > 30 and i.area > 30 and j.width > 5 and i.width > 5 and j.height > 5 and i.height > 5 \
                   and j.height + j.pos_y < im_height - 6 and j.pos_y > im_height / 100 * 10 \
                   and i.height + i.pos_y < im_height - 6 and i.pos_y > im_height / 100 * 10:

                    sm = img_for_count[:, i.pos_x:i.pos_x + i.width].sum(axis=0)
                    sm = enumerate(sm, start=i.pos_x)
                    sm = sorted(sm, key=lambda x: x[1])

                    matched_none_char.append(i)
                    matched_none_char.append(j)
                    if i.width >= 4:
                        col_for_fill = 4
                    else:
                        col_for_fill = 2

                    for g in range(col_for_fill):
                        for k in range(im_height - 1):
                            if img_thresh[k, sm[g][0]] == 255 and img_thresh[k + 1, sm[g][0]] == 0:
                                for l in range(k, im_height - 1):
                                    if img_thresh[l + 1, sm[g][0]] == 0:
                                        img_thresh[l + 1, sm[g][0]] = 255
                                    else:
                                        break
                                break

    return img_thresh


def replace_last_first(img_thresh, list_of_psb_chr, nmbr_of_chrs):
    list_of_psb_chr = sorted(list_of_psb_chr, key=lambda obj: obj.center_x)
    sm = 0
    im_height = img_thresh.shape[0]

    # find mean spacing between symbols
    for i in range(1, nmbr_of_chrs - 1):
        sm += list_of_psb_chr[i + 1].pos_x - (list_of_psb_chr[i].pos_x + list_of_psb_chr[i].width)
    mean_spacing = sm / (nmbr_of_chrs - 1)

    # check if first symbol is redundant
    cur_symb = list_of_psb_chr[0]
    next_symb = list_of_psb_chr[1]
    if abs(next_symb.pos_x - (cur_symb.pos_x + cur_symb.width)) > mean_spacing or \
            abs(next_symb.pos_y - cur_symb.pos_y) > im_height / 100 * 3 or \
            abs((next_symb.pos_y + next_symb.height) - (cur_symb.pos_y + cur_symb.height)) > im_height / 100 * 3:
        img_thresh[:, :cur_symb.pos_x + cur_symb.width] = 0
        if nmbr_of_chrs <= 8:
            return img_thresh
        else:
            # check if last symbol is redundant
            cur_symb = list_of_psb_chr[-1]
            prev_symb = list_of_psb_chr[-2]
            if cur_symb.center_x - prev_symb.center_x > mean_spacing and \
                    abs(cur_symb.center_y - prev_symb.center_y) > im_height / 100 * 10:
                img_thresh[:, cur_symb.pos_x:] = 0

    return img_thresh


def separate_stick(img_thresh, list_of_psb_chr, none_chars):
    list_for_separate_line = []

    for i in none_chars:
        if len(list_of_psb_chr) > 2 and i.area > list_of_psb_chr[-1].area * 1.5 and \
                abs(i.center_y - list_of_psb_chr[-2].center_y) < 10 < i.height:

            if list_of_psb_chr[-2].width / i.width > 0.33:
                list_for_separate_line.append(int(i.center_x))

            elif list_of_psb_chr[-2].width / i.width > 0.25:
                step = i.width / 3
                list_for_separate_line.append(int(i.pos_x + step))
                list_for_separate_line.append(int(i.pos_x + step * 2))

            else:
                step = i.width / 4
                list_for_separate_line.append(int(i.pos_x + step))
                list_for_separate_line.append(int(i.pos_x + step * 2))
                list_for_separate_line.append(int(i.pos_x + step * 3))

    for i in list_for_separate_line:
        img_thresh[:, i] = 0

    return img_thresh


def del_blue(img_orig, list_of_psb_chr, img_thresh):

    blue = [np.array([86, 31, 4]), np.array([255, 150, 150])]
    mask = cv2.inRange(img_orig, blue[0], blue[1])
    output = cv2.bitwise_and(img_orig, img_orig, mask=mask)

    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    _, output = cv2.threshold(output, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    list_of_psb_chr = sorted(list_of_psb_chr, key=lambda x: x.center_x)
    frst_char = list_of_psb_chr[0]
    roi = output[frst_char.pos_y:frst_char.pos_y + frst_char.height, frst_char.pos_x:frst_char.pos_x + frst_char.width]

    _, roi = cv2.threshold(roi, 1, 1, cv2.THRESH_BINARY)

    if np.mean(roi) < 0.3:
        img_thresh[:, :frst_char.pos_x + frst_char.width] = 0

    return img_thresh


def histogram_kray_fill(img_thresh, img_grayscale, list_of_psb_chr):
    im = img_thresh
    height, width = im.shape
    hist = []
    frame = 5

    # calculating histogram
    for x in range(width - 1):
        column = im[0:width - 1, x:x + 1]
        hist.append(np.sum(column) / len(column))

    hist = np.array(hist)
    border = np.mean(hist[10:-11]) + 5

    indx_min = 0
    indx_max = 0
    for x in range(int(width / frame) - 1):
        mn = np.amin(hist[x * frame + 1:(x + 1) * frame + 1])
        new_mn = np.amin(hist[(x + 1) * frame + 1:(x + 2) * frame + 1])

        if mn < new_mn:
            indx_min = np.argmin(hist[x * frame + 1:(x + 1) * frame + 1]) + x * frame + 1
            break

    first_max = np.max(hist[:11])
    for x in range(int(width / frame)):
        mx = np.amax(hist[x * frame:(x + 1) * frame])
        new_mx = np.amax(hist[(x + 1) * frame:(x + 2) * frame])
        indx_max = np.argmax(hist[x * frame:(x + 1) * frame]) + x * frame

        if mx > new_mx and ((mx > border and indx_max > indx_min) or (mx > first_max + 10 and indx_max < indx_min)):
            indx_max = np.argmax(hist[x * frame:x * frame + frame]) + x * frame

            for i, kd in zip(list_of_psb_chr, range(len(list_of_psb_chr))):
                if i.center_x > indx_max > i.pos_x:
                    indx_max = i.pos_x - 2
                    break

            if indx_max < 0:
                indx_max = 1

            img_thresh = img_thresh[:, indx_max:]
            img_grayscale = img_grayscale[:, indx_max:]
            break

    shift = indx_max

    im = img_thresh
    height, width = im.shape
    hist = []

    for x in range(width - 1):
        column = im[0:width - 1, x:x + 1]
        hist.append(np.sum(column) / len(column))

    hist = np.array(hist)

    indx_min = 0
    indx_max = 0
    for x in range(int(width / frame), 1, -1):
        mn = np.amin(hist[(x - 1) * frame:x * frame])
        new_mn = np.amin(hist[(x - 2) * frame:(x - 1) * frame])

        if mn < new_mn:
            indx_min = np.argmin(hist[(x - 1) * frame:x * frame]) + (x - 1) * frame
            break

    first_max = np.max(hist[-12:])
    for x in range(int(width / frame), 1, -1):
        mx = np.max(hist[(x - 1) * frame:x * frame])
        new_mx = np.max(hist[(x - 2) * frame:(x - 1) * frame])
        indx_max = np.argmax(hist[(x - 1) * frame:x * frame]) + (x - 1) * frame

        if mx > new_mx and ((mx > border and indx_max < indx_min) or (mx > first_max + 10 and indx_max > indx_min)):
            indx_max = np.argmax(hist[(x - 1) * frame:x * frame]) + (x - 1) * frame + 3

            for i, kd in zip(list_of_psb_chr, range(len(list_of_psb_chr))):
                if i.center_x < indx_max + shift < i.pos_x + i.width:
                    indx_max = i.pos_x + i.width + 2 - shift
                    break

            if indx_max > width - 1:
                indx_max = width - 2

            img_thresh = img_thresh[:, :indx_max]
            img_grayscale = img_grayscale[:, :indx_max]
            break

    return img_thresh, img_grayscale


# PreProc[70, 3, 15, 0.1, 1.5] find_psbl_char
# Br[100, 4, 15, 0.15, 1] find_psbl_chr_4broken

def rotate(img_orig, list_of_matching_chars):
    list_of_matching_chars.sort(key=lambda char: char.center_x)

    # calculate the center point of the plate
    flt_plate_center_x = (list_of_matching_chars[0].center_x + list_of_matching_chars[
        len(list_of_matching_chars) - 1].center_x) / 2.0
    flt_plate_center_y = (list_of_matching_chars[0].center_y + list_of_matching_chars[
        len(list_of_matching_chars) - 1].center_y) / 2.0
    # This is the probable centeral point of this plate.
    pt_plate_center = flt_plate_center_x, flt_plate_center_y

    # calculate correction angle of plate region
    flt_opposite = list_of_matching_chars[-1].center_y - list_of_matching_chars[0].center_y
    flt_hypotenuse = DetectChars.distance_between_chars(list_of_matching_chars[0], list_of_matching_chars[-1])
    try:
        flt_correction_angle_in_rad = math.asin(flt_opposite / flt_hypotenuse)
    except:
        flt_correction_angle_in_rad = 0
    flt_correction_angle_in_deg = flt_correction_angle_in_rad * (180.0 / math.pi)

    # final steps are to perform the actual rotation
    # get the rotation matrix for our calculated correction angle
    # The first poin tis the point of rotaion or center,theta and scaling factor
    rotation_matrix = cv2.getRotationMatrix2D(tuple(pt_plate_center), flt_correction_angle_in_deg, 1.0)

    height, width = img_orig.shape  # unpack original image width and height

    img_rotated = cv2.warpAffine(img_orig, rotation_matrix, (width, height))  # rotate the entire image

    return img_rotated


def preprocess_plate_without_croping(img_orig):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))

    im = clahe.apply(cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY))

    img_thresh = half_thresh(im)
    img_thresh = kray_fill(img_thresh, full_filling=True)

    list_of_possible_chars, number_of_chars, _ = DetectChars.find_psbl_chr(img_thresh, [70, 3, 15, 0.1, 1.5])

    if number_of_chars > 7:
        img_thresh = replace_last_first(img_thresh, list_of_possible_chars, number_of_chars)
        img_thresh = del_blue(img_orig, list_of_possible_chars, img_thresh)

    list_of_possible_chars, _, _ = DetectChars.find_psbl_chr(img_thresh, [70, 3, 15, 0.1, 1.5])

    img_thresh = draw_top_line(list_of_possible_chars, img_thresh)
    img_thresh = draw_btm_line(list_of_possible_chars, img_thresh)

    list_of_possible_chars, number_of_chars, none_char = DetectChars.find_psbl_chr(img_thresh, [70, 3, 15, 0.1, 1.5])

    if number_of_chars >= 2:
        img_thresh = separate_stick(img_thresh, list_of_possible_chars, none_char)

    _, _, none_char = DetectChars.find_psbl_chr(img_thresh, [100, 4, 15, 0.15, 1])

    img_for_count = img_thresh.copy()
    _, img_for_count = cv2.threshold(img_for_count, 1, 1, cv2.THRESH_BINARY_INV)
    img_thresh = matching_broken_chars(img_thresh, none_char)

    return img_thresh
