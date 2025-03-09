# Allowable libraries:
# - Python 3.10.12
# - Pillow 10.0.0
# - numpy 1.25.2
# - OpenCV 4.6.0 (with opencv-contrib-python-headless 4.6.0.66)

# To activate image processing, uncomment the following imports:
from PIL import Image
import numpy as np
import cv2

class Agent:
    def __init__(self):
        self.rotation_angles = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 
                              202.5, 225, 247.5, 270, 292.5, 315, 337.5]
        self.flip_codes = [0, 1, -1]
        self.scale_factors = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
        self.rule_weights = {
            'constant_row': 1.3,
            'quantitative_progression': 1.3,
            'figure_addition': 1.2,
            'distribution_three': 1.3
        }

    def preprocess(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        return thresh

    def detect_transformations(self, src, target):
        transformations = []
        for angle in self.rotation_angles:
            rotated = self.rotate_image(src, angle)
            similarity = cv2.matchTemplate(rotated, target, cv2.TM_CCOEFF_NORMED)[0][0]
            if similarity > self.adaptive_threshold(0.65):
                transformations.append(('rotate', angle, similarity))

        for flip in self.flip_codes:
            flipped = cv2.flip(src, flip)
            similarity = cv2.matchTemplate(flipped, target, cv2.TM_CCOEFF_NORMED)[0][0]
            if similarity > self.adaptive_threshold(0.6):
                transformations.append(('flip', flip, similarity))

        for scale in self.scale_factors:
            scaled = cv2.resize(src, None, fx=scale, fy=scale)
            similarity = cv2.matchTemplate(scaled, target, cv2.TM_CCOEFF_NORMED)[0][0]
            if similarity > self.adaptive_threshold(0.55):
                transformations.append(('scale', scale, similarity))

        xor_diff = cv2.bitwise_xor(src, target)
        contours, _ = cv2.findContours(xor_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            total_area = sum(cv2.contourArea(c) for c in contours)
            if 50 < total_area < (src.size * 0.25):
                transformations.append(('shape_change', xor_diff, 0.7))
                
        return transformations

    def generate_predictions(self, base_img, transformations):
        predictions = []
        for transform in sorted(transformations, key=lambda x: x[2], reverse=True):
            try:
                if transform[0] == 'rotate':
                    img = self.rotate_image(base_img, transform[1])
                elif transform[0] == 'flip':
                    img = cv2.flip(base_img, transform[1])
                elif transform[0] == 'scale':
                    img = cv2.resize(base_img, None, fx=transform[1], fy=transform[1])
                elif transform[0] == 'shape_change':
                    img = cv2.bitwise_xor(base_img, transform[1])
                predictions.append(img)
                if len(predictions) >= 8:
                    break
            except:
                continue
        return predictions

    def rotate_image(self, img, angle):
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(img, M, (w, h))

    def adaptive_threshold(self, base_score):
        return max(0.5, min(0.8, base_score * 0.95))

    def Solve(self, problem):
        try:
            A = self.preprocess(problem.figures["A"].visualFilename)
            B = self.preprocess(problem.figures["B"].visualFilename)
            C = self.preprocess(problem.figures["C"].visualFilename)

            row_transforms = self.detect_transformations(A, B)
            col_transforms = self.detect_transformations(A, C)

            best_score = -1
            best_answer = -1
            
            for i in range(1, 7):
                option = str(i)
                if option not in problem.figures:
                    continue
                target = self.preprocess(problem.figures[option].visualFilename)
                total_score = 0

                row_preds = self.generate_predictions(B, row_transforms)
                col_preds = self.generate_predictions(C, col_transforms)
                cross_preds = [cv2.bitwise_and(r, c) for r in row_preds for c in col_preds]

                for pred in row_preds + col_preds + cross_preds:
                    similarity = cv2.matchTemplate(pred, target, cv2.TM_CCOEFF_NORMED)[0][0]
                    total_score += similarity * self.rule_weights.get('figure_addition', 1.0)

                if total_score > best_score:
                    best_score = total_score
                    best_answer = i

            return best_answer if best_score > self.adaptive_threshold(0.6) else -1
        except Exception as e:
            return -1