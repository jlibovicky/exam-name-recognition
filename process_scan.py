#!/usr/bin/env python3

import argparse
import base64
import logging
from unidecode import unidecode

import cv2
import numpy as np
from pdf2image import convert_from_path
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
from torchvision import transforms


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


ALPHABET = "ABCDEFGIHJKLMNOPQRSTUVWXYZ "
CHARS = {char: i for i, char in enumerate(ALPHABET)}


def find_aruco_markers(img):
    parameters = cv2.aruco.DetectorParameters()
    marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(marker_dict, parameters)
    corners, ids, _ = detector.detectMarkers(img)
    assert len(corners) == len(ids) == 3
    sorted_corners = sort_corners(corners, ids)
    non_zero_id = next(idx[0] for idx in ids if idx != [0])
    return sorted_corners, non_zero_id


def sort_corners(corners, ids):
    corners = [corner[0][0] for corner in corners]
    zero_points = sorted(
        [corner for corner, id in zip(corners, ids) if id == [0]],
        key=lambda x: x[0])
    other_points = sorted(
        [corner for corner, id in zip(corners, ids) if id != [0]],
        key=lambda x: x[0])
    return zero_points + other_points


CHARBOX_H = 57
CHARBOX_W = 35
LINE_X = 344
LINE_1_Y = 313
LINE_2_Y = 394


def highlight_charboxes(img):
    for i in range(28):
        start_x = LINE_X + i * (CHARBOX_W + 4) + (i // 3)
        cv2.rectangle(img, (start_x, LINE_1_Y),
                      (start_x + CHARBOX_W, LINE_1_Y + CHARBOX_H),
                      (0, 0, 255), 2)
        cv2.rectangle(img, (start_x, LINE_2_Y),
                      (start_x + CHARBOX_W, LINE_2_Y + CHARBOX_H),
                      (0, 0, 255), 2)


def extract_charboxes(img):
    first_name = []
    surname = []

    for i in range(28):
        start_x = LINE_X + i * (CHARBOX_W + 4) + (i // 3)
        first_name.append(
            img[LINE_1_Y:LINE_1_Y + CHARBOX_H, start_x:start_x + CHARBOX_W])
        surname.append(
            img[LINE_2_Y:LINE_2_Y + CHARBOX_H, start_x:start_x + CHARBOX_W])

    first_name = [normalize_char(box) for box in first_name]
    surname = [normalize_char(box) for box in surname]
    return first_name, surname


def normalize_char(char_img):
    char_img = char_img[3:-2, 3:-2]
    scale = 28 / char_img.shape[0]
    char_img = 255 - cv2.resize(char_img, (0, 0), fx=scale, fy=scale)

    square = np.zeros((28, 28), dtype=np.uint8)
    # Put the image into horizontal center
    start_x = (28 - char_img.shape[1]) // 2
    square[:, start_x:start_x + char_img.shape[1]] = char_img
    return square


class Classifier:
    def __init__(self, model_path):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        self.classifier = torch.load(
            model_path, map_location=torch.device('cpu'))

    @torch.no_grad()
    def classify(self, first_name, surname):
        first_name = [self.data_transform(char_img) for char_img in first_name]
        surname = [self.data_transform(char_img) for char_img in surname]

        as_tensor = torch.stack(first_name + surname, dim=0)
        output = self.classifier(as_tensor)
        log_prob_first_name = F.log_softmax(output[:28], dim=1).numpy()
        log_prob_surname = F.log_softmax(output[28:], dim=1).numpy()

        return log_prob_first_name, log_prob_surname


def score_string(log_probs, string):
    string += " "
    score = 0
    for log_prob, char in zip(log_probs, string):
        score += log_prob[CHARS[char]]
    return score / len(string)


def score_student_list(fist_name_log_probs, surname_log_probs, student_list):
    """For each student, how likely this their name."""
    scores = []
    for _, norm_names in student_list:
        first_name_score = max(
            score_string(fist_name_log_probs, first_name)
            for first_name, _ in norm_names)
        surname_score = max(
            score_string(surname_log_probs, surname)
            for _, surname in norm_names)

        scores.append((first_name_score + surname_score) / 2)
    return scores


def norm_name(name):
    normalized_name = unidecode(name).upper()
    # Split on first space only
    surname, first_name = normalized_name.split(" ", 1)
    return first_name, surname


def load_student_list(student_list_path):
    names = []
    with open(student_list_path, encoding="utf-8") as f_list:
        for line in f_list:
            name = line.strip()
            # Remove everything after comma (titles)
            name = name.split(",")[0]
            names.append((name, [norm_name(name)]))

            if "CH" in name.upper():
                names[-1][1].append(
                    norm_name(name.upper().replace("CH", "H")))
    return names


def to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode("ascii")


def img_to_array(img):
    return np.array(img).astype(np.uint8).reshape(-1).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_pdf", help="Reference PDF file.")
    parser.add_argument("scanned_pdf", help="Scanned PDF file.")
    parser.add_argument("student_list", help="Student list, name per line.")
    parser.add_argument(
        "--ocr-model", default="shufflenet.pt", help="OCR model path.")
    parser.add_argument(
        "--html-report", default="report.html", help="Path to HTML report.")
    parser.add_argument(
        "--character-csv", default="characters.csv", help="Path to CSV file.")
    args = parser.parse_args()

    logging.info("Loading OCR model.")
    classifier = Classifier(args.ocr_model)

    logging.info("Loading student list.")
    student_list = load_student_list(args.student_list)

    logging.info("Loading reference PDF for calibration.")
    original = np.array(
        convert_from_path(args.reference_pdf, grayscale=True)[0])
    ref_corners, _ = find_aruco_markers(original)
    assert len(ref_corners) == 3
    #for corner in ref_corners:
    #    corner = corner.astype(int)
    #    cv2.rectangle(original, tuple(corner[0][0]), tuple(corner[0][2]), (0, 255, 0), 2)

    #highlight_charboxes(original)

    #cv2.imwrite("original.png", original)

    logging.info("Loading scanned PDF (takes a while).")
    scanned_ppm = convert_from_path(
        args.scanned_pdf, grayscale=True, use_pdftocairo=False)
    scanned_cairo = convert_from_path(
        args.scanned_pdf, grayscale=True, use_pdftocairo=True)
    student_signiture_matching = []
    first_name_imgs = []
    surname_imgs = []
    first_character_imgs = []
    surname_character_imgs = []

    logging.info("Processing pages.")
    failed_pages = 0
    for page_no, (page_ppm, page_cairo) in enumerate(zip(scanned_ppm, scanned_cairo)):
        page = np.array(page_ppm)
        try:
            corners, page_id = find_aruco_markers(page)
        except:
            try:
                page = np.array(page_cairo)
                corners, page_id = find_aruco_markers(page)
            except:
                cv2.imwrite(f"failed_page{page_no:03d}.png", page)
                logging.warning("Failed to find markers on page %d.", page_no + 1)
                failed_pages += 1
                continue

        if page_id != 1:
            continue
        transform = cv2.getAffineTransform(
            np.stack(corners), np.stack(ref_corners))
        new_page = cv2.warpAffine(
            page, transform, (original.shape[1], original.shape[0]))
        highlight_charboxes(new_page)

        first_name_boxes, surname_boxes = extract_charboxes(new_page)
        first_name_imgs.append(np.concatenate(first_name_boxes, axis=1))
        surname_imgs.append(np.concatenate(surname_boxes, axis=1))

        first_character_imgs.append(first_name_boxes)
        surname_character_imgs.append(surname_boxes)

        first_name_log_probs, surname_log_probs = classifier.classify(
            first_name_boxes, surname_boxes)
        student_signiture_matching.append(
            score_student_list(
                first_name_log_probs, surname_log_probs, student_list))

    if failed_pages > 0:
        logging.warning(
            "Failed to process %d pages of %d.",
            failed_pages, len(scanned_ppm))

    logging.info("Matching students to signitures.")
    img_ids, student_ids = linear_sum_assignment(
        student_signiture_matching, maximize=True)

    logging.info("Generating HTML report.")
    with open(args.html_report, "w") as f_html:
        print("<html><head><meta charset='utf-8'></head><body>", file=f_html)
        print("<table>", file=f_html)

        for img_id, student_id in zip(img_ids, student_ids):
            print("<tr>", file=f_html)
            print(f"<td>{student_list[student_id][0]}</td>", file=f_html)
            print(f"<td><img src='data:image/png;base64,{to_base64(surname_imgs[img_id])}'/><br />", file=f_html)
            print(f"<img src='data:image/png;base64,{to_base64(first_name_imgs[img_id])}'/></td>", file=f_html)
            print("</tr>", file=f_html)

        print("</table>", file=f_html)
        print("</body></html>", file=f_html)

    logging.info("Character images as CSV.")
    with open(args.character_csv, "w") as f_csv:
        for img_id, student_id in zip(img_ids, student_ids):
            surname, first_name = student_list[student_id][0].split(" ", 1)
            surname += " "
            first_name += " "

            for surname_char, img in zip(
                    surname, surname_character_imgs[img_id]):
                img_list = img_to_array(img)
                img_line = ",".join(map(str, img_list))
                print(f"{surname_char},{img_line}", file=f_csv)

    # Print the matched students
    for student_id in student_ids:
        print(student_list[student_id][0])

    logging.info("Done.")


if  __name__ == "__main__":
    main()
