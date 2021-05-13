import copy
import statistics
import cv2
import numpy as np
import imutils
from imutils import contours
import four_point


def extract_size(item):
    (_, _, width, heigth) = cv2.boundingRect(item)
    return (width, heigth)


def calc_sizes_avg(data):
    sizes = list(map(extract_size, data))
    filtered_sizes = list(
        filter(lambda item: is_between(item[0]/item[1], 1, 0.2), sizes))
    means_sizes = list(map(statistics.median_high, list(zip(*filtered_sizes))))
    return tuple(means_sizes)


def is_between(value, limit, error_margin=2):
    return limit - error_margin <= value <= limit + error_margin


def show_contours(cnts, background):
    image = copy.deepcopy(background)
    cv2.drawContours(image, cnts, -1, 255, -1)
    cv2.imshow("show", image)
    cv2.waitKey(0)


def filter_contours(all_contours, mean_width, mean_height):
    question_cnts = []
    for contour in all_contours:
        (width, height) = extract_size(contour)
        if is_between(width, mean_width) and is_between(height, mean_height):
            question_cnts.append(contour)
    return question_cnts


def sum_pixels(contour, thresh):
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mask = cv2.bitwise_and(thresh, thresh, mask=mask)
    return cv2.countNonZero(mask)


def apply_threshold(cuted_image):
    thresh = cv2.threshold(
        cuted_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return thresh


def get_contours(image):
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts


def get_doc_contour(all_contours, accuracy_tax):
    if len(all_contours) > 0:
        all_sorted_contours = sorted(
            all_contours,
            key=cv2.contourArea,
            reverse=True
        )
    for contour in all_sorted_contours:
        perimeter_contour = cv2.arcLength(contour, True)
        epsilon = accuracy_tax * perimeter_contour
        approximated_contour = cv2.approxPolyDP(
            contour, epsilon, True)
        if len(approximated_contour) == 4:
            doc_cnt = approximated_contour
            return doc_cnt



def main(image_path="omr2.png",
         num_lines=5,
         num_options=5,
         accuracy_tax=0.02):

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    all_contours = cnts = get_contours(edged)

    doc_cnt = get_doc_contour(all_contours, accuracy_tax)    

    paper = four_point.four_point_transform(image, doc_cnt.reshape(4, 2))

    warped = four_point.four_point_transform(gray, doc_cnt.reshape(4, 2))

    show_contours([], image)
    show_contours([], paper)

    thresh = apply_threshold(warped)

    show_contours([], thresh)

    cnts = get_contours(thresh)

    show_contours(cnts, paper)

    (mean_width, mean_height) = calc_sizes_avg(cnts)

    valid_bubbles_options = filter_contours(cnts, mean_width, mean_height)

    show_contours(valid_bubbles_options, paper)
    valid_bubbles_options.reverse()

    for (_, i) in enumerate(np.arange(0, len(valid_bubbles_options), num_lines)):
        cnts = contours.sort_contours(
            valid_bubbles_options[i: i + num_options])[0]
        list_num_pixels = list(
            map(lambda item: sum_pixels(item, thresh), cnts))
        mean_num_pixels = statistics.mean(list_num_pixels)
        counter = sum(
            [1 for item in list_num_pixels if item > mean_num_pixels])
        print(counter)


main()
# main("form.png", num_lines=6, num_options=6)
