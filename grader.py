import statistics
import cv2
import numpy as np
import imutils
import four_point
from imutils import contours


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
    cv2.drawContours(background, cnts, -1, 255, -1)
    cv2.imshow("show", background)
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


def get_contours(thresh):
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts


def main(IMAGE_PATH="omr2.png",
         NUM_LINES=5,
         NUM_OPTIONS=5,
         APPROXION_ACCURACY_TAX=0.02):

    image = cv2.imread(IMAGE_PATH)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    all_contours = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = imutils.grab_contours(all_contours)
    doc_cnt = None

    if len(all_contours) > 0:
        all_sorted_contours = sorted(
            all_contours,
            key=cv2.contourArea,
            reverse=True
        )
        for contour in all_sorted_contours:
            perimeter_contour = cv2.arcLength(contour, True)
            epsilon = APPROXION_ACCURACY_TAX * perimeter_contour
            approximated_contour = cv2.approxPolyDP(
                contour, epsilon, True)
            if len(approximated_contour) == 4:
                doc_cnt = approximated_contour
                break

    # paper = four_point.four_point_transform(image, doc_cnt.reshape(4, 2))
    warped = four_point.four_point_transform(gray, doc_cnt.reshape(4, 2))
    thresh = apply_threshold(warped)
    cnts = get_contours(thresh)
    (mean_width, mean_height) = calc_sizes_avg(cnts)
    valid_bubbles_options = filter_contours(cnts, mean_width, mean_height)

    for (_, i) in enumerate(np.arange(0, len(valid_bubbles_options), NUM_LINES)):
        cnts = contours.sort_contours(
            valid_bubbles_options[i: i + NUM_OPTIONS])[0]
        list_num_pixels = list(
            map(lambda item: sum_pixels(item, thresh), cnts))
        mean_num_pixels = statistics.mean(list_num_pixels)
        counter = sum(
            [1 for item in list_num_pixels if item > mean_num_pixels])
        print(counter)


main()
