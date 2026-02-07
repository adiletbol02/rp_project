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

    def preprocess_image(self, img_input, params=None, debug_dir=None):
        # Default robust parameters (You can override these via the 'params' argument)
        cfg = {
            'target_height': 800,       # Fixed height for consistency
            'blur_d': 9,                # Bilateral filter diameter
            'blur_sigma': 75,           # Bilateral filter color/space sigma
            'thresh_block': 31,         # Block size for adaptive threshold (must be odd)
            'thresh_c': 5,             # Constant subtracted from mean (The "Sensitivity" knob)
            'morph_op': 'close',        # 'open' removes noise, 'close' fills gaps (safer for thin lines)
            'morph_kernel': 2,          # Size of the morphological kernel
            'morph_iter': 1             # Number of times to run morphology
        }
        if params:
            cfg.update(params)


        # Handle input: if string, load path. If array, use directly.
        if isinstance(img_input, str):
            img = cv2.imread(img_input, cv2.IMREAD_GRAYSCALE)
        else:
            # Assume it's already a numpy array (e.g. from buffer)
            if len(img_input.shape) == 3:
                img = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
            else:
                img = img_input

        # 1. INTELLIGENT RESIZE (Robustness)
        # Standardize height to 800px so blur/threshold kernels work consistently
        h, w = img.shape
        scale = cfg['target_height'] / h
        new_w = int(w * scale)
        img_resized = cv2.resize(img, (new_w, cfg['target_height']))

        if debug_dir: cv2.imwrite(f"{debug_dir}/1_resized.png", img_resized)

        # 2. DENOISE (Bilateral is better than Gaussian)
        # Bilateral filter smooths flat regions but keeps text edges crisp
        denoised = cv2.bilateralFilter(img_resized, cfg['blur_d'], cfg['blur_sigma'], cfg['blur_sigma'])

        if debug_dir: cv2.imwrite(f"{debug_dir}/2_denoised.png", denoised)

        # 3. THRESHOLDING
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            cfg['thresh_block'],
            cfg['thresh_c']
        )

        # 4. MORPHOLOGICAL CLEANUP (Robustness)
        # Removes tiny specks that thresholding missed
        kernel = np.ones((cfg['morph_kernel'], cfg['morph_kernel']), np.uint8)

        if cfg['morph_op'] == 'open':
            clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=cfg['morph_iter'])
        elif cfg['morph_op'] == 'close':
            clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=cfg['morph_iter'])
        else:
            clean = binary

        if debug_dir: cv2.imwrite(f"{debug_dir}/3_binarized_clean.png", clean)




        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return clean

        initial_boxes = [cv2.boundingRect(c) for c in contours]
        areas = [b[2] * b[3] for b in initial_boxes]
        top_3_area = np.mean(sorted(areas, reverse=True)[:3])

        # Create a mask to keep only significant blobs
        clean_mask = np.zeros_like(clean)
        for i, b in enumerate(initial_boxes):
            area = b[2] * b[3]
            # If it's big enough, draw it onto our clean mask
            if area > (top_3_area / 15) or (b[2] > 2.5 * b[3] and area > top_3_area / 20):
                cv2.drawContours(clean_mask, contours, i, 255, -1)


        

        # 5. DESKEWING
        coords = np.column_stack(np.where(clean_mask > 0))
        if len(coords) == 0:
            return clean_mask # Return empty if no text found

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = clean_mask.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(clean_mask, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        if debug_dir: cv2.imwrite(f"{debug_dir}/4_deskewed.png", rotated)

        return rotated


    def find_and_filter_contours(self, clean_binary, debug_dir=None):
        """
        Input is already noise-filtered and deskewed.
        Goal: Group related blobs (like '=' or 'i') into single character boxes.
        """
        # 1. Get contours from the cleaned image
        contours, _ = cv2.findContours(clean_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours]
        
        if not boxes: 
            return []

        # 2. ITERATIVE MERGE LOGIC
        # We must group vertically stacked parts (equals signs, division dots, 'i' dots)
        while True:
            merged = False
            new_boxes = []
            skip_indices = set()
            
            # Sort Left-to-Right to make neighbor checking predictable
            boxes.sort(key=lambda b: b[0])

            for i in range(len(boxes)):
                if i in skip_indices: continue
                x1, y1, w1, h1 = boxes[i]
                merged_this_iter = False

                for j in range(i + 1, len(boxes)):
                    if j in skip_indices: continue
                    x2, y2, w2, h2 = boxes[j]

                    # A. Vertical Stacking Check (Crucial for '=' and 'i')
                    # Calculate horizontal overlap
                    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    # If they share 40% of their width, they are likely part of the same char
                    is_stacked = (x_overlap > 0.4 * min(w1, w2))

                    # B. Tight Overlap Check (In case blobs touch or intersect)
                    xi1, yi1 = max(x1, x2), max(y1, y2)
                    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
                    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    union_area = (w1 * h1) + (w2 * h2) - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0

                    if is_stacked or iou > 0.3:
                        # Create the new bounding box that encompasses both
                        nx = min(x1, x2)
                        ny = min(y1, y2)
                        nw = max(x1 + w1, x2 + w2) - nx
                        nh = max(y1 + h1, y2 + h2) - ny
                        
                        new_boxes.append((nx, ny, nw, nh))
                        skip_indices.add(j)
                        merged = True
                        merged_this_iter = True
                        break

                if not merged_this_iter:
                    new_boxes.append(boxes[i])

            boxes = new_boxes
            if not merged: 
                break

        # Final sort ensures your equation string (e.g., "2+2") is in the right order
        boxes.sort(key=lambda b: b[0])
        return boxes

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









