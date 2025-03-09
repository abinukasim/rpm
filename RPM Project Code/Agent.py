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
        self.flip_codes = [0, 1, -1]  # Horizontal, Vertical, Both
        self.scale_factors = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
        
        # Pattern weights for scoring - tuned for C problem sets
        self.rule_weights = {
            'constant_row': 1.6,          # Same pattern applies across a row
            'constant_column': 1.6,       # Same pattern applies down a column
            'quantitative_progression': 1.5,  # Values change in a consistent way
            'figure_addition': 1.3,       # Elements are added across figures
            'distribution_three': 1.5,    # Pattern distributed across 3 figures
            'shape_change': 1.4,         # Shape modifications
            'arithmetic_operation': 1.7,  # Logical operations (AND, OR, XOR)
            'alternating_pattern': 1.6,   # Patterns that alternate (common in C problems)
            'shape_progression': 1.5      # Progressive changes to shapes
        }

    def preprocess(self, img_path):
        """Preprocess image to binary form"""
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        return thresh

    def adaptive_threshold(self, base_score):
        """Adjust threshold based on context"""
        return max(0.5, min(0.8, base_score * 0.95))

    def detect_transformations(self, src, target):
        """Detect possible transformations between source and target images"""
        transformations = []
        
        # Check for rotations
        for angle in self.rotation_angles:
            try:
                rotated = self.rotate_image(src, angle)
                similarity = cv2.matchTemplate(rotated, target, cv2.TM_CCOEFF_NORMED)[0][0]
                if similarity > self.adaptive_threshold(0.65):
                    transformations.append(('rotate', angle, similarity))
            except:
                continue
        
        # Check for flips
        for flip in self.flip_codes:
            try:
                flipped = cv2.flip(src, flip)
                similarity = cv2.matchTemplate(flipped, target, cv2.TM_CCOEFF_NORMED)[0][0]
                if similarity > self.adaptive_threshold(0.6):
                    transformations.append(('flip', flip, similarity))
            except:
                continue
        
        # Check for scaling
        for scale in self.scale_factors:
            try:
                scaled = cv2.resize(src, None, fx=scale, fy=scale)
                similarity = cv2.matchTemplate(scaled, target, cv2.TM_CCOEFF_NORMED)[0][0]
                if similarity > self.adaptive_threshold(0.55):
                    transformations.append(('scale', scale, similarity))
            except:
                continue
        
        # Check for XOR (shape changes)
        try:
            # Ensure images are the same size
            h, w = src.shape
            target_resized = cv2.resize(target, (w, h))
            
            xor_diff = cv2.bitwise_xor(src, target_resized)
            contours, _ = cv2.findContours(xor_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                total_area = sum(cv2.contourArea(c) for c in contours)
                if 50 < total_area < (src.size * 0.25):
                    transformations.append(('shape_change', xor_diff, 0.7))
        except:
            pass
                
        return transformations

    def rotate_image(self, img, angle):
        """Rotate an image by the given angle"""
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(img, M, (w, h))

    def generate_predictions(self, base_img, transformations):
        """Generate predictions by applying transformations to the base image"""
        predictions = []
        
        for transform in sorted(transformations, key=lambda x: x[2], reverse=True):
            try:
                if transform[0] == 'rotate':
                    img = self.rotate_image(base_img, transform[1])
                elif transform[0] == 'flip':
                    img = cv2.flip(base_img, transform[1])
                elif transform[0] == 'scale':
                    img = cv2.resize(base_img, None, fx=transform[1], fy=transform[1])
                    # Resize back to original dimensions for easier comparison
                    h, w = base_img.shape
                    img = cv2.resize(img, (w, h))
                elif transform[0] == 'shape_change':
                    h, w = base_img.shape
                    xor_img = cv2.resize(transform[1], (w, h))
                    img = cv2.bitwise_xor(base_img, xor_img)
                else:
                    continue
                
                predictions.append(img)
                if len(predictions) >= 8:  # Limit number of predictions
                    break
            except:
                continue
        
        return predictions

    def solve_2x2(self, problem):
        """Solve a 2x2 Raven's Progressive Matrix problem"""
        try:
            # Load and preprocess figures
            A = self.preprocess(problem.figures["A"].visualFilename)
            B = self.preprocess(problem.figures["B"].visualFilename)
            C = self.preprocess(problem.figures["C"].visualFilename)
            
            # Detect transformations
            row_transforms = self.detect_transformations(A, B)
            col_transforms = self.detect_transformations(A, C)
            
            # Score answer choices
            best_score = -1
            best_answer = -1
            
            for i in range(1, 7):  # 2x2 problems have 6 answer choices
                option = str(i)
                if option not in problem.figures:
                    continue
                    
                target = self.preprocess(problem.figures[option].visualFilename)
                total_score = 0
                
                # Generate predictions
                row_preds = self.generate_predictions(C, row_transforms)
                col_preds = self.generate_predictions(B, col_transforms)
                
                # Create cross predictions (combinations of row and column transformations)
                cross_preds = []
                for r in row_preds:
                    for c in col_preds:
                        try:
                            h, w = r.shape
                            c_resized = cv2.resize(c, (w, h))
                            cross_preds.append(cv2.bitwise_and(r, c_resized))
                        except:
                            continue
                
                # Score against all predictions
                for pred in row_preds + col_preds + cross_preds:
                    try:
                        h, w = pred.shape
                        target_resized = cv2.resize(target, (w, h))
                        similarity = cv2.matchTemplate(pred, target_resized, cv2.TM_CCOEFF_NORMED)[0][0]
                        total_score += similarity * self.rule_weights.get('figure_addition', 1.0)
                    except:
                        continue
                
                if total_score > best_score:
                    best_score = total_score
                    best_answer = i
            
            return best_answer if best_score > self.adaptive_threshold(0.6) else -1
        except Exception as e:
            return -1

    def analyze_pattern_distribution(self, figures):
        """Analyze how elements are distributed in the matrix"""
        pattern_metrics = {}
        
        # Check for image coverage/density in each cell
        for key, img in figures.items():
            white_pixels = np.sum(img > 0)
            total_pixels = img.size
            density = white_pixels / total_pixels
            pattern_metrics[key + '_density'] = density
        
        # Check for alternating patterns in rows
        for row_keys in [["A", "B", "C"], ["D", "E", "F"], ["G", "H"]]:
            densities = [pattern_metrics[k + '_density'] for k in row_keys if k in pattern_metrics]
            if len(densities) >= 2:
                alternating = all(densities[i] > densities[i+1] for i in range(0, len(densities)-1, 2)) or \
                              all(densities[i] < densities[i+1] for i in range(0, len(densities)-1, 2))
                pattern_metrics['alternating_' + ''.join(row_keys)] = alternating
        
        return pattern_metrics

    def detect_arithmetic_operations(self, figures):
        """Detect arithmetic operations (addition, subtraction) between figures"""
        operations = []
        
        key_pairs = [
            (("A", "B"), "C"),  # A op B = C
            (("A", "D"), "G"),  # A op D = G
            (("B", "E"), "H"),  # B op E = H
            (("D", "E"), "F"),  # D op E = F
        ]
        
        for (src1_key, src2_key), result_key in key_pairs:
            if src1_key not in figures or src2_key not in figures or result_key not in figures:
                continue
                
            src1 = figures[src1_key]
            src2 = figures[src2_key]
            result = figures[result_key]
            
            # Ensure same dimensions for operations
            h, w = src1.shape
            src2_resized = cv2.resize(src2, (w, h))
            result_resized = cv2.resize(result, (w, h))
            
            # Test AND operation
            and_result = cv2.bitwise_and(src1, src2_resized)
            and_similarity = cv2.matchTemplate(and_result, result_resized, cv2.TM_CCOEFF_NORMED)[0][0]
            
            # Test OR operation
            or_result = cv2.bitwise_or(src1, src2_resized)
            or_similarity = cv2.matchTemplate(or_result, result_resized, cv2.TM_CCOEFF_NORMED)[0][0]
            
            # Test XOR operation
            xor_result = cv2.bitwise_xor(src1, src2_resized)
            xor_similarity = cv2.matchTemplate(xor_result, result_resized, cv2.TM_CCOEFF_NORMED)[0][0]
            
            # Determine the most likely operation
            ops = [('AND', and_similarity), ('OR', or_similarity), ('XOR', xor_similarity)]
            best_op = max(ops, key=lambda x: x[1])
            
            if best_op[1] > 0.7:  # Strong match threshold
                operations.append((src1_key, src2_key, result_key, best_op[0], best_op[1]))
        
        return operations

    def detect_shape_progression(self, figures):
        """Detect shape progression/changes across figures"""
        progressions = []
        
        # Define sequences to check
        sequences = [
            ["A", "B", "C"],
            ["D", "E", "F"],
            ["G", "H"],
            ["A", "D", "G"],
            ["B", "E", "H"],
            ["C", "F"]
        ]
        
        for seq in sequences:
            if not all(k in figures for k in seq):
                continue
                
            # Check for consistent changes
            shape_metrics = []
            for key in seq:
                img = figures[key]
                contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Calculate contour metrics
                num_contours = len(contours)
                total_area = sum(cv2.contourArea(c) for c in contours) if contours else 0
                total_perimeter = sum(cv2.arcLength(c, True) for c in contours) if contours else 0
                
                shape_metrics.append({
                    'key': key,
                    'num_contours': num_contours,
                    'area': total_area,
                    'perimeter': total_perimeter
                })
            
            # Check for numerical progression in metrics
            for metric in ['num_contours', 'area', 'perimeter']:
                values = [m[metric] for m in shape_metrics]
                
                # Linear progression
                if len(values) >= 3:
                    diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
                    if all(abs(diffs[0] - d) / (diffs[0] + 0.001) < 0.1 for d in diffs):
                        progressions.append(('linear', seq, metric, values, diffs[0]))
                
                # Geometric progression
                if len(values) >= 3 and all(v > 0 for v in values):
                    ratios = [values[i+1] / values[i] for i in range(len(values)-1)]
                    if all(abs(ratios[0] - r) / ratios[0] < 0.1 for r in ratios):
                        progressions.append(('geometric', seq, metric, values, ratios[0]))
        
        return progressions

    def solve_3x3(self, problem):
        """Solve a 3x3 Raven's Progressive Matrix problem with enhanced pattern recognition"""
        try:
            # Load and preprocess all figures
            figures = {}
            for key in ["A", "B", "C", "D", "E", "F", "G", "H"]:
                figures[key] = self.preprocess(problem.figures[key].visualFilename)
            
            # Analyze row patterns (horizontal progression)
            row1_AB = self.detect_transformations(figures["A"], figures["B"])
            row1_BC = self.detect_transformations(figures["B"], figures["C"])
            row2_DE = self.detect_transformations(figures["D"], figures["E"])
            row2_EF = self.detect_transformations(figures["E"], figures["F"])
            row3_GH = self.detect_transformations(figures["G"], figures["H"])
            
            # Analyze column patterns (vertical progression)
            col1_AD = self.detect_transformations(figures["A"], figures["D"])
            col1_DG = self.detect_transformations(figures["D"], figures["G"])
            col2_BE = self.detect_transformations(figures["B"], figures["E"])
            col2_EH = self.detect_transformations(figures["E"], figures["H"])
            col3_CF = self.detect_transformations(figures["C"], figures["F"])
            
            # Advanced pattern analysis for 3x3 problems
            pattern_distribution = self.analyze_pattern_distribution(figures)
            arithmetic_operations = self.detect_arithmetic_operations(figures)
            shape_progressions = self.detect_shape_progression(figures)
            
            # Generate predictions for the missing figure (position I)
            
            # 1. Row-based predictions (apply patterns from rows 1-2 to row 3)
            row_preds = []
            # Apply row1 B→C pattern to H
            row_preds.extend(self.generate_predictions(figures["H"], row1_BC))
            # Apply row2 E→F pattern to H
            row_preds.extend(self.generate_predictions(figures["H"], row2_EF))
            # Continue row3 pattern G→H→?
            row_preds.extend(self.generate_predictions(figures["H"], row3_GH))
            
            # 2. Column-based predictions (apply patterns from cols 1-2 to col 3)
            col_preds = []
            # Apply col1 D→G pattern to F
            col_preds.extend(self.generate_predictions(figures["F"], col1_DG))
            # Apply col2 E→H pattern to F
            col_preds.extend(self.generate_predictions(figures["F"], col2_EH))
            # Continue col3 pattern C→F→?
            col_preds.extend(self.generate_predictions(figures["F"], col3_CF))
            
            # 3. Diagonal-based predictions
            diag_preds = []
            # Apply A→E pattern to H
            diag_AE = self.detect_transformations(figures["A"], figures["E"])
            diag_preds.extend(self.generate_predictions(figures["H"], diag_AE))
            # Apply E→I pattern to a copy of I following A→E
            if "E" in figures and "I" in figures:
                diag_preds.extend(self.generate_predictions(figures["E"], diag_AE))
            
            # 4. Logic-based predictions (combinations)
            logic_preds = []
            try:
                h, w = figures["H"].shape
                
                # Generate predictions based on detected arithmetic operations
                for (src1_key, src2_key, result_key, op, similarity) in arithmetic_operations:
                    if op == 'AND' and src1_key == "G" and src2_key == "F":
                        f_resized = cv2.resize(figures["F"], (w, h))
                        logic_preds.append(cv2.bitwise_and(figures["H"], f_resized))
                    elif op == 'OR' and src1_key == "G" and src2_key == "F":
                        f_resized = cv2.resize(figures["F"], (w, h))
                        logic_preds.append(cv2.bitwise_or(figures["H"], f_resized))
                    elif op == 'XOR' and src1_key == "G" and src2_key == "F":
                        f_resized = cv2.resize(figures["F"], (w, h))
                        logic_preds.append(cv2.bitwise_xor(figures["H"], f_resized))
                
                # Additional operations specifically for 3x3 matrix patterns
                
                # Test for ABC pattern: if A+B=C, then G+H=?
                if "A" in figures and "B" in figures and "C" in figures:
                    a_resized = cv2.resize(figures["A"], (w, h))
                    b_resized = cv2.resize(figures["B"], (w, h))
                    c_resized = cv2.resize(figures["C"], (w, h))
                    
                    # If A OR B is close to C
                    ab_or = cv2.bitwise_or(a_resized, b_resized)
                    if cv2.matchTemplate(ab_or, c_resized, cv2.TM_CCOEFF_NORMED)[0][0] > 0.7:
                        logic_preds.append(cv2.bitwise_or(figures["G"], figures["H"]))
                    
                    # If A AND B is close to C
                    ab_and = cv2.bitwise_and(a_resized, b_resized)
                    if cv2.matchTemplate(ab_and, c_resized, cv2.TM_CCOEFF_NORMED)[0][0] > 0.7:
                        logic_preds.append(cv2.bitwise_and(figures["G"], figures["H"]))
                    
                    # If A XOR B is close to C
                    ab_xor = cv2.bitwise_xor(a_resized, b_resized)
                    if cv2.matchTemplate(ab_xor, c_resized, cv2.TM_CCOEFF_NORMED)[0][0] > 0.7:
                        logic_preds.append(cv2.bitwise_xor(figures["G"], figures["H"]))
                
                # Test for row-column synthesis: if C = A op B and G = A op D, then I = G op H or C op F?
                if all(k in figures for k in ["A", "B", "C", "D", "F", "G", "H"]):
                    g_resized = cv2.resize(figures["G"], (w, h))
                    h_resized = cv2.resize(figures["H"], (w, h))
                    c_resized = cv2.resize(figures["C"], (w, h))
                    f_resized = cv2.resize(figures["F"], (w, h))
                    
                    logic_preds.append(cv2.bitwise_and(g_resized, h_resized))
                    logic_preds.append(cv2.bitwise_or(g_resized, h_resized))
                    logic_preds.append(cv2.bitwise_xor(g_resized, h_resized))
                    logic_preds.append(cv2.bitwise_and(c_resized, f_resized))
                    logic_preds.append(cv2.bitwise_or(c_resized, f_resized))
                    logic_preds.append(cv2.bitwise_xor(c_resized, f_resized))
            except:
                pass
            
            # Combine all predictions
            all_preds = row_preds + col_preds + diag_preds + logic_preds
            
            # Score answer choices
            best_score = -1
            best_answer = -1
            
            for i in range(1, 9):  # 3x3 problems have 8 answer choices
                option = str(i)
                if option not in problem.figures:
                    continue
                    
                target = self.preprocess(problem.figures[option].visualFilename)
                option_score = 0
                
                # Score against row predictions (patterns across rows)
                for pred in row_preds:
                    try:
                        h, w = pred.shape
                        target_resized = cv2.resize(target, (w, h))
                        similarity = cv2.matchTemplate(pred, target_resized, cv2.TM_CCOEFF_NORMED)[0][0]
                        option_score += similarity * self.rule_weights.get('constant_row', 1.0)
                    except:
                        continue
                        
                # Score against column predictions (patterns down columns)
                for pred in col_preds:
                    try:
                        h, w = pred.shape
                        target_resized = cv2.resize(target, (w, h))
                        similarity = cv2.matchTemplate(pred, target_resized, cv2.TM_CCOEFF_NORMED)[0][0]
                        option_score += similarity * self.rule_weights.get('constant_column', 1.0)
                    except:
                        continue
                        
                # Score against diagonal and logic predictions
                for pred in diag_preds + logic_preds:
                    try:
                        h, w = pred.shape
                        target_resized = cv2.resize(target, (w, h))
                        similarity = cv2.matchTemplate(pred, target_resized, cv2.TM_CCOEFF_NORMED)[0][0]
                        option_score += similarity * self.rule_weights.get('distribution_three', 1.0)
                    except:
                        continue
                
                # Additional check: directly compare pattern consistency
                # Check if H→target maintains the same pattern as B→C
                h_to_ans = self.detect_transformations(figures["H"], target)
                for h_trans in h_to_ans:
                    for b_trans in row1_BC:
                        if h_trans[0] == b_trans[0]:  # Same transformation type
                            option_score += h_trans[2] * b_trans[2] * 1.5  # Higher weight for pattern consistency
                
                # Check if F→target maintains the same pattern as D→G
                f_to_ans = self.detect_transformations(figures["F"], target)
                for f_trans in f_to_ans:
                    for d_trans in col1_DG:
                        if f_trans[0] == d_trans[0]:  # Same transformation type
                            option_score += f_trans[2] * d_trans[2] * 1.5  # Higher weight for pattern consistency
                
                # Pattern distribution similarity
                if 'alternating_ABC' in pattern_distribution and pattern_distribution['alternating_ABC']:
                    # Check if this option continues the alternating pattern
                    h_density = pattern_distribution.get('H_density', 0)
                    target_density = np.sum(target > 0) / target.size
                    
                    # If G, H are alternating, target should continue that pattern
                    g_density = pattern_distribution.get('G_density', 0)
                    if (g_density > h_density and h_density < target_density) or \
                       (g_density < h_density and h_density > target_density):
                        option_score += 1.2  # Bonus for continuing alternating pattern
                
                # Shape progression consistency
                for prog_type, seq, metric, values, rate in shape_progressions:
                    if len(seq) >= 2 and seq[-1] in ["H", "F"]:
                        # Calculate expected value based on progression
                        contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if metric == 'num_contours':
                            actual = len(contours)
                        elif metric == 'area':
                            actual = sum(cv2.contourArea(c) for c in contours) if contours else 0
                        elif metric == 'perimeter':
                            actual = sum(cv2.arcLength(c, True) for c in contours) if contours else 0
                        
                        if prog_type == 'linear':
                            expected = values[-1] + rate
                        elif prog_type == 'geometric':
                            expected = values[-1] * rate
                        
                        # Score based on how close actual is to expected
                        progression_similarity = 1 - min(1, abs(actual - expected) / (expected + 0.001))
                        option_score += progression_similarity * 1.4  # Higher weight for progression patterns
                
                # For each arithmetic operation detected, check if the answer continues the pattern
                for src1_key, src2_key, result_key, op, similarity in arithmetic_operations:
                    if (src1_key == "H" and src2_key == "F") or (src1_key == "F" and src2_key == "H"):
                        h, w = figures["H"].shape
                        f_resized = cv2.resize(figures["F"], (w, h))
                        target_resized = cv2.resize(target, (w, h))
                        
                        if op == 'AND':
                            expected = cv2.bitwise_and(figures["H"], f_resized)
                        elif op == 'OR':
                            expected = cv2.bitwise_or(figures["H"], f_resized)
                        elif op == 'XOR':
                            expected = cv2.bitwise_xor(figures["H"], f_resized)
                        
                        op_similarity = cv2.matchTemplate(expected, target_resized, cv2.TM_CCOEFF_NORMED)[0][0]
                        option_score += op_similarity * 1.6  # Higher weight for arithmetic operations
                
                if option_score > best_score:
                    best_score = option_score
                    best_answer = i
            
            return best_answer if best_score > self.adaptive_threshold(0.6) else -1
        except Exception as e:
            return -1

    def Solve(self, problem):
        """Main solving method for all Raven's Progressive Matrix problems"""
        try:
            if problem.problemType == "3x3":
                return self.solve_3x3(problem)
            else:  # 2x2 problem
                return self.solve_2x2(problem)
        except Exception as e:
            return -1