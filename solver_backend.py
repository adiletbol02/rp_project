import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import sympy as sp
import os

# --- CONFIGURATION ---
IMG_SIZE = 28

# Complete Class Map (Ensure this matches your training)
CLASS_MAP = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '+': 10, '-': 11, '=': 12, 'times': 13, 'div': 14, '/': 15, '.': 16, 'pm': 17,
    '(': 18, ')': 19, '[': 20, ']': 21, '{': 22, '}': 23, ',': 24, '!': 25, 'prime': 26, '|': 27,
    '<': 28, '>': 29, 'leq': 30, 'geq': 31, 'neq': 32, 'exists': 33, 'forall': 34, 'in': 35,
    'rightarrow': 36, 'infty': 37, 'ldots': 38,
    'sqrt': 39, 'int': 40, 'sum': 41, 'lim': 42, 'log': 43, 'sin': 44, 'cos': 45, 'tan': 46,
    'alpha': 47, 'beta': 48, 'gamma': 49, 'Delta': 50, 'theta': 51, 'lambda': 52, 'mu': 53,
    'pi': 54, 'sigma': 55, 'phi': 56,
    'x': 57, 'X': 58, 'y': 59, 'z': 60,
    'A': 61, 'b': 62, 'C': 63, 'd': 64, 'e': 65, 'f': 66, 'G': 67, 'H': 68,
    'i': 69, 'j': 70, 'k': 71, 'l': 72, 'M': 73, 'N': 74, 'o': 75, 'p': 76,
    'q': 77, 'R': 78, 'S': 79, 'T': 80, 'u': 81, 'v': 82, 'w': 83
}
ID_TO_LABEL = {v: k for k, v in CLASS_MAP.items()}

class EquationSolver:
    def __init__(self, model_path):
        print(f"Loading model from {model_path}...")
        try:
            self.model = load_model(model_path)
            self.model_loaded = True
            print("Model loaded successfully.")
        except Exception as e:
            self.model_loaded = False
            print(f"Error loading model: {e}")

    def preprocess_image(self, img_input, debug_dir=None):
        # Handle input types
        if isinstance(img_input, str):
            img = cv2.imread(img_input, cv2.IMREAD_GRAYSCALE)
        else:
            if len(img_input.shape) == 3:
                img = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
            else:
                img = img_input

        # 1. INTELLIGENT RESIZE
        # We resize height to 800px so our kernel sizes (51, 15) always behave the same
        h, w = img.shape
        target_height = 800
        scale = target_height / h
        new_w = int(w * scale)
        img_resized = cv2.resize(img, (new_w, target_height))

        # 2. DENOISE (Bilateral)
        denoised = cv2.bilateralFilter(img_resized, 11, 75, 75)

        # 3. THRESHOLDING (The Fix for Hollow Strokes)
        # BlockSize 51, C 15 ensures thick pen strokes don't turn white inside
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 71, 15)

        # 4. MORPHOLOGICAL CLEANUP
        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # 5. DESKEWING
        coords = np.column_stack(np.where(clean > 0))
        if len(coords) == 0:
            return clean # Return empty if no text found

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = clean.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(clean, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated

    def find_and_filter_contours(self, binary_img):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours]
        
        if not boxes: return []

        # --- MERGE LOGIC ---
        while True:
            merged = False
            new_boxes = []
            skip_indices = set()
            boxes.sort(key=lambda b: b[0])

            for i in range(len(boxes)):
                if i in skip_indices: continue
                x1, y1, w1, h1 = boxes[i]
                merged_this_iter = False

                for j in range(i + 1, len(boxes)):
                    if j in skip_indices: continue
                    x2, y2, w2, h2 = boxes[j]

                    # Overlap Checks
                    xi1, yi1 = max(x1, x2), max(y1, y2)
                    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
                    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    area1, area2 = w1 * h1, w2 * h2
                    
                    overlap_ratio = inter_area / min(area1, area2) if min(area1, area2) > 0 else 0
                    
                    # Vertical Stacking Check
                    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    is_stacked = (x_overlap > 0.4 * min(w1, w2))

                    if overlap_ratio > 0.5 or is_stacked:
                        nx, ny = min(x1, x2), min(y1, y2)
                        nw, nh = max(x1 + w1, x2 + w2) - nx, max(y1 + h1, y2 + h2) - ny
                        new_boxes.append((nx, ny, nw, nh))
                        skip_indices.add(j)
                        merged = True
                        merged_this_iter = True
                        break

                if not merged_this_iter:
                    new_boxes.append(boxes[i])

            boxes = new_boxes
            if not merged: break

        # --- FILTERING LOGIC ---
        areas = [b[2] * b[3] for b in boxes]
        median_area = np.median(areas)
        final_boxes = []
        
        for b in boxes:
            w, h = b[2], b[3]
            area = w * h
            # Robustness: Don't delete wide, thin lines (Minus signs)
            is_minus_like = (w > 2.5 * h) and (area > median_area / 15)
            
            if area > median_area / 5 or is_minus_like:
                final_boxes.append(b)

        final_boxes.sort(key=lambda b: b[0])
        return final_boxes

    def extract_and_predict(self, binary_img, boxes):
        if not boxes: return []
        batch_images = []

        for (x, y, w, h) in boxes:
            roi = binary_img[y:y+h, x:x+w]
            
            # Pad to square
            max_dim = max(w, h)
            pad_w, pad_h = (max_dim - w) // 2, (max_dim - h) // 2
            roi_padded = cv2.copyMakeBorder(roi, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
            
            # Resize
            roi_resized = cv2.resize(roi_padded, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            
            # Normalize
            roi_norm = roi_resized.astype('float32') / 255.0
            roi_norm = np.expand_dims(roi_norm, axis=-1)
            batch_images.append(roi_norm)

        # Batch Prediction
        batch_input = np.array(batch_images)
        pred_probs = self.model.predict(batch_input, verbose=0)
        pred_indices = np.argmax(pred_probs, axis=1)
        
        return [ID_TO_LABEL[idx] for idx in pred_indices]

    def disambiguate_symbols(self, labels, boxes):
        """
        Strict Contextual Disambiguation Logic:
        - Math operators are the PRIMARY driver for disambiguation
        - Shape/height heuristics are secondary fallbacks
        - Resolves ambiguities: x/times, 5/s, 1/l/|, 0/o
        """
        if len(boxes) < 1: return labels
        refined = list(labels)
        
        # Stats
        heights = [b[3] for b in boxes]
        avg_height = np.median(heights)
        centroids_y = [(b[1] + b[3] / 2) for b in boxes]

        # Define math operator set for context checking
        math_operators = ['+', '-', '=', '*', '/', 'times', 'div', 'pm']

        for i in range(len(refined)):
            lbl = refined[i]
            x, y, w, h = boxes[i]
            cy = centroids_y[i]
            prev_lbl = refined[i-1] if i > 0 else None
            next_lbl = refined[i+1] if i < len(refined) - 1 else None
            
            # Context flags (computed once per iteration)
            prev_is_op = prev_lbl in math_operators if prev_lbl else False
            next_is_op = next_lbl in math_operators if next_lbl else False
            prev_is_digit = prev_lbl and prev_lbl.isdigit() if prev_lbl else False
            next_is_digit = next_lbl and next_lbl.isdigit() if next_lbl else False
            digit_context = prev_is_digit or next_is_digit
            
            # 1. 'x' vs 'times' (multiplication)
            # RULE: If x is next to a math operator, it is a VARIABLE
            if lbl in ['x', 'times', 'X']:
                if prev_is_op or next_is_op:
                    # Math context → x is a variable
                    refined[i] = 'x'
                else:
                    # No operator context, check other heuristics
                    is_between_nums = prev_is_digit and next_is_digit
                    is_small = h < (0.85 * avg_height)
                    is_floating = False
                    if prev_lbl or next_lbl:
                        neighbor_cy = centroids_y[i-1] if prev_lbl else centroids_y[i+1]
                        if cy < (neighbor_cy - 0.1 * h): 
                            is_floating = True

                    if is_between_nums or is_small or is_floating:
                        refined[i] = 'times'
                    else:
                        refined[i] = 'x'

            # 2. '5' vs 's' / 'S'
            # RULE: If '5' is next to a math operator or digit, it is a DIGIT
            elif lbl in ['5', 's', 'S']:
                if prev_is_op or next_is_op or digit_context:
                    # Math/digit context → treat as digit '5'
                    refined[i] = '5'
                else:
                    # Check for variable context (sin, cos, etc.)
                    is_trig = (next_lbl in ['i', 'I']) or (prev_lbl in ['o', 'O', 'c', 'C'])
                    is_math = (prev_lbl in ['+', '-', '=']) or (next_lbl in ['+', '-', '='])
                    if is_trig: 
                        refined[i] = 's'
                    elif is_math: 
                        refined[i] = '5'
                    else:
                        # Default: treat as '5' if uncertain
                        refined[i] = '5'

            # 3. '1' vs 'l' vs 'I' vs '|'
            # RULE: If next to digit/operator, prefer '1'; if in variable context, prefer 'l'
            elif lbl in ['1', 'l', '|', 'I', 'i']:
                if digit_context or prev_is_op or next_is_op:
                    # Digit/math context → '1' is most likely
                    refined[i] = '1'
                elif h > 1.4 * avg_height:
                    # Very tall → vertical bar '|'
                    refined[i] = '|'
                elif next_lbl in ['n', 'N', 'o', 'O']:
                    # Word context (e.g., "in", "on") → 'l'
                    refined[i] = 'l'
                else:
                    # Default: '1'
                    refined[i] = '1'

            # 4. '0' vs 'o'
            # RULE: If next to digit/operator, prefer '0'; if in variable context, prefer 'o'
            elif lbl in ['0', 'o', 'O']:
                if digit_context or prev_is_op or next_is_op:
                    # Digit/math context → '0'
                    refined[i] = '0'
                elif prev_lbl in ['c', 'C', 'l', 'L']:
                    # Letter context (e.g., "co", "lo") → 'o'
                    refined[i] = 'o'
                else:
                    # Default: '0' (more common in equations)
                    refined[i] = '0'

        return refined

    def parse_and_solve(self, labels):
        map_op = {
            'times': '*', 'div': '/', 'plus': '+', '-': '-',
            'pm': '+', '=': '=', 'x': 'x', 'y': 'y', 'z': 'z', 'pi': 'pi',
            'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'sqrt': 'sqrt'
        }

        equation_str = ""
        for lbl in labels:
            equation_str += map_op.get(lbl, lbl)

        if equation_str.endswith('='): equation_str = equation_str[:-1]

        transformations = (standard_transformations + (implicit_multiplication_application,))
        try:
            # Case 1: Simple Expression "2+2"
            if '=' not in equation_str:
                expr = parse_expr(equation_str, transformations=transformations)
                res = float(expr)
                # Nice formatting for integers
                result_str = f"{int(res)}" if res.is_integer() else f"{res:.2f}"
                return equation_str, result_str
            
            # Case 2: Equation "2x=10"
            else:
                parts = equation_str.split('=')
                if len(parts) != 2: return equation_str, "Error: Multiple '='"
                lhs_str, rhs_str = parts
                
                # Handle "2+2=" empty right side
                if not rhs_str.strip():
                    expr = parse_expr(lhs_str, transformations=transformations)
                    res = float(expr)
                    result_str = f"{int(res)}" if res.is_integer() else f"{res:.2f}"
                    return lhs_str, result_str

                lhs = parse_expr(lhs_str, transformations=transformations)
                rhs = parse_expr(rhs_str, transformations=transformations)
                solution = sp.solve(lhs - rhs)
                return equation_str, f"{solution}"

        except Exception as e:
            return equation_str, f"Error: {str(e)}"

    def solve_image(self, image_buffer):
        """
        The main method called by app.py.
        1. Reads buffer -> Image
        2. Pipeline (Preprocess -> Box -> Predict -> Logic -> Solve)
        3. Returns (Image, EquationString, ResultString)
        """
        if not self.model_loaded:
            return None, "Model Error", "Check .h5 file"

        # Read Streamlit buffer to CV2 image
        file_bytes = np.asarray(bytearray(image_buffer.read()), dtype=np.uint8)
        original = cv2.imdecode(file_bytes, 1)

        # 1. Preprocess
        processed = self.preprocess_image(original)

        # 2. Segment
        boxes = self.find_and_filter_contours(processed)
        if not boxes:
            return processed, "No Content", "Could not find text"

        # 3. Predict
        labels = self.extract_and_predict(processed, boxes)

        # 4. Disambiguate (Smart Logic)
        labels = self.disambiguate_symbols(labels, boxes)

        # 5. Solve
        final_eq, result = self.parse_and_solve(labels)

        # 6. Visualize (Draw on image for the UI)
        vis_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis_img, labels[i], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return vis_img, final_eq, result





