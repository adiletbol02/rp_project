import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import sympy as sp

# --- CONFIGURATION ---
IMG_SIZE = 28
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
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def preprocess_image(self, img_input, debug_dir=None):
        """
        ROBUST UPDATES:
        1. Resize to fixed height (800px) for consistent kernel behavior.
        2. Bilateral Filter (keeps edges sharp, removes noise).
        3. Morphological Cleanup (removes tiny dots).
        """
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
        target_height = 800
        scale = target_height / h
        new_w = int(w * scale)
        img_resized = cv2.resize(img, (new_w, target_height))
        
        if debug_dir: cv2.imwrite(f"{debug_dir}/1_resized.png", img_resized)

        # 2. DENOISE (Bilateral is better than Gaussian)
        # Bilateral filter smooths flat regions but keeps text edges crisp
        denoised = cv2.bilateralFilter(img_resized, 9, 75, 75)
        
        if debug_dir: cv2.imwrite(f"{debug_dir}/2_denoised.png", denoised)

        # 3. THRESHOLDING
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 95, 17)

        # 4. MORPHOLOGICAL CLEANUP (Robustness)
        # Removes tiny specks that thresholding missed
        kernel = np.ones((2, 2), np.uint8)
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        if debug_dir: cv2.imwrite(f"{debug_dir}/3_binarized_clean.png", clean)

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

        if debug_dir: cv2.imwrite(f"{debug_dir}/4_deskewed.png", rotated)

        return rotated

    def find_and_filter_contours(self, binary_img, debug_dir=None):
        """
        ROBUST UPDATES:
        1. Added "Minus Sign Protection" logic so thin dashes aren't deleted.
        """
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if debug_dir:
            raw_vis = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(raw_vis, contours, -1, (0, 0, 255), 1)
            cv2.imwrite(f"{debug_dir}/5_raw_contours.png", raw_vis)

        boxes = [cv2.boundingRect(c) for c in contours]
        if not boxes: return []

        # --- MERGE LOGIC (Iterative) ---
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

                    # Overlap
                    xi1 = max(x1, x2)
                    yi1 = max(y1, y2)
                    xi2 = min(x1 + w1, x2 + w2)
                    yi2 = min(y1 + h1, y2 + h2)
                    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    area1, area2 = w1 * h1, w2 * h2
                    
                    overlap_ratio = inter_area / min(area1, area2) if min(area1, area2) > 0 else 0

                    # Stacking (Vertical alignment)
                    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    is_stacked = (x_overlap > 0.4 * min(w1, w2))

                    if overlap_ratio > 0.5 or is_stacked:
                        nx, ny = min(x1, x2), min(y1, y2)
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
            if not merged: break

        # --- SMARTER FILTERING (Robustness) ---
        if not boxes: return []
        areas = [b[2] * b[3] for b in boxes]
        median_area = np.median(areas)

        final_boxes = []
        for b in boxes:
            w, h = b[2], b[3]
            area = w * h
            
            # EXCEPTION: Minus signs are wide and short. Don't delete them!
            is_minus_like = (w > 2.5 * h) and (area > median_area / 15)

            # Keep if area is normal OR if it looks like a minus sign
            if area > median_area / 5 or is_minus_like:
                final_boxes.append(b)

        final_boxes.sort(key=lambda b: b[0])

        if debug_dir:
            vis_box = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
            for (x,y,w,h) in final_boxes:
                cv2.rectangle(vis_box, (x,y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite(f"{debug_dir}/6_filtered_boxes.png", vis_box)

        return final_boxes

    def extract_and_predict(self, binary_img, boxes, debug_dir=None):
        """
        ROBUST UPDATES:
        1. Implemented BATCH PREDICTION (Run all symbols at once).
        """
        if not boxes: return []
        
        batch_images = []

        for i, (x, y, w, h) in enumerate(boxes):
            roi = binary_img[y:y+h, x:x+w]

            # Square Padding
            max_dim = max(w, h)
            pad_w = (max_dim - w) // 2
            pad_h = (max_dim - h) // 2
            roi_padded = cv2.copyMakeBorder(roi, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)

            # Resize to Model Input
            roi_resized = cv2.resize(roi_padded, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

            if debug_dir:
                cv2.imwrite(f"{debug_dir}/7_cnn_input_{i}.png", roi_resized)

            # Normalize
            roi_norm = roi_resized.astype('float32') / 255.0
            roi_norm = np.expand_dims(roi_norm, axis=-1)
            batch_images.append(roi_norm)

        # --- BATCH PREDICT (Speed) ---
        batch_input = np.array(batch_images) # Shape: (N, 28, 28, 1)
        pred_probs = self.model.predict(batch_input, verbose=0)
        pred_indices = np.argmax(pred_probs, axis=1)
        
        predictions = [ID_TO_LABEL[idx] for idx in pred_indices]
        return predictions

    def parse_and_solve(self, labels):
        # ... (Same logic as before, ensure map_op is robust) ...
        map_op = {
            'times': '*', 'div': '/', 'plus': '+', '-': '-',
            'pm': '+', '=': '=', 'x': 'x', 'y': 'y', 'z': 'z',
            'pi': 'pi' # Add more if your model supports them
        }

        equation_str = ""
        for lbl in labels:
            if lbl in map_op:
                equation_str += map_op[lbl]
            else:
                equation_str += lbl

        if equation_str.endswith('='):
            equation_str = equation_str[:-1]

        print(f"Constructed String: {equation_str}")
        transformations = (standard_transformations + (implicit_multiplication_application,))

        try:
            if '=' not in equation_str:
                expr = parse_expr(equation_str, transformations=transformations)
                return f"{equation_str} = {float(expr):.2f}"
            else:
                parts = equation_str.split('=')
                if len(parts) != 2: return "Error: Invalid '=' usage"
                lhs_str, rhs_str = parts
                if not rhs_str.strip():
                    expr = parse_expr(lhs_str, transformations=transformations)
                    return f"{lhs_str} = {float(expr):.2f}"
                
                lhs = parse_expr(lhs_str, transformations=transformations)
                rhs = parse_expr(rhs_str, transformations=transformations)
                solution = sp.solve(lhs - rhs)
                return f"Solution: {solution}"
        except Exception as e:
            return f"Error solving: {e}"

    def disambiguate_symbols(self, labels, boxes):
        """
        Refines predictions based on geometry (size/position) and context (neighbors).
        Handles: 'x' vs 'times', '5' vs 's', '1' vs 'l' vs '|', '0' vs 'o'
        """
        if len(boxes) < 1: return labels
        
        refined = list(labels)
        
        # 1. Calculate Global Statistics
        heights = [b[3] for b in boxes]
        avg_height = np.median(heights)
        
        # Calculate Y-Centroids (vertical centers) for floating checks
        centroids_y = [(b[1] + b[3] / 2) for b in boxes]

        for i in range(len(refined)):
            lbl = refined[i]
            x, y, w, h = boxes[i]
            cy = centroids_y[i] # Center Y of current box

            # Identify Neighbors
            prev_lbl = refined[i-1] if i > 0 else None
            next_lbl = refined[i+1] if i < len(refined) - 1 else None
            
            # --- LOGIC GROUP 1: 'x' vs 'times' ---
            if lbl in ['x', 'times', 'X']:
                # Rule A: Strictly between numbers -> 'times' (e.g., "2 x 3")
                is_between_nums = (prev_lbl and prev_lbl.isdigit()) and (next_lbl and next_lbl.isdigit())
                
                # Rule B: Size check (Times is usually smaller)
                is_small = h < (0.85 * avg_height)
                
                # Rule C: Floating Check (Times floats above baseline)
                # Compare my center to the average center of the whole equation (rough approx)
                # or better, compare to neighbors if available.
                is_floating = False
                if prev_lbl or next_lbl:
                    neighbor_cy = centroids_y[i-1] if prev_lbl else centroids_y[i+1]
                    # If my center is significantly higher (smaller Y value) than neighbor
                    if cy < (neighbor_cy - 0.1 * h): 
                        is_floating = True

                if is_between_nums:
                    refined[i] = 'times'
                elif is_small or is_floating:
                    refined[i] = 'times'
                else:
                    refined[i] = 'x' # Default to variable if big and on baseline (e.g., "2x")

            # --- LOGIC GROUP 2: '5' vs 's' ---
            elif lbl in ['5', 's', 'S']:
                # Rule A: Look for "sin", "cos", "sec" patterns
                # If next is 'i' (sin) or prev is 'o' (cos), it's 's'
                is_trig_context = (next_lbl in ['i', 'I']) or (prev_lbl in ['o', 'O', 'c', 'C'])
                
                # Rule B: If sandwiched by operators, it's likely 5 (e.g., "+ 5 +")
                is_math_context = (prev_lbl in ['+', '-', '=', 'times']) or (next_lbl in ['+', '-', '=', 'times'])

                if is_trig_context:
                    refined[i] = 's'
                elif is_math_context:
                    refined[i] = '5'
                # Fallback: maintain original prediction if unsure

            # --- LOGIC GROUP 3: '1' vs 'l' vs '|' ---
            elif lbl in ['1', 'l', '|', 'I', 'i']:
                # Rule A: Absolute Value bars are usually taller than the median (e.g. |x|)
                if h > 1.4 * avg_height:
                    refined[i] = '|'
                # Rule B: Logarithms (ln, log)
                elif next_lbl in ['n', 'N', 'o', 'O']:
                    refined[i] = 'l' # Lowercase L for log/ln
                # Rule C: If solitary or near math, likely 1
                else:
                    refined[i] = '1'

            # --- LOGIC GROUP 4: '0' vs 'o' ---
            elif lbl in ['0', 'o', 'O']:
                # Rule A: "cos", "cot", "log"
                if prev_lbl in ['c', 'C', 'l', 'L']:
                    refined[i] = 'o'
                # Rule B: Math context
                elif (prev_lbl and prev_lbl.isdigit()) or (next_lbl and next_lbl.isdigit()):
                    refined[i] = '0'

        return refined

    def run(self, image_source, output_path=None, debug=False):
        """
        Accepts image path OR image bytes/array.
        """
        debug_dir = None
        if debug:
            debug_dir = "debug_output"
            os.makedirs(debug_dir, exist_ok=True)
            print(f"Debug mode on. Saving steps to /{debug_dir}...")

        # 1. Process
        processed_img = self.preprocess_image(image_source, debug_dir=debug_dir)

        # 2. Segment
        boxes = self.find_and_filter_contours(processed_img, debug_dir=debug_dir)

        # 3. Predict
        labels = self.extract_and_predict(processed_img, boxes, debug_dir=debug_dir)
        
        # --- RESTORED STEP: DISAMBIGUATE ---
        # Refines labels (e.g., changes 'x' to 'times' based on context)
        labels = self.disambiguate_symbols(labels, boxes)
        # -----------------------------------

        # 4. Solve
        solution_text = self.parse_and_solve(labels)

        # 5. Visualize
        vis_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        for i, (x, y, w, h) in enumerate(boxes):
            # Draw box
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Draw label
            cv2.putText(vis_img, labels[i], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw Solution Text
        cv2.putText(vis_img, f"Sol: {solution_text}", (10, vis_img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if output_path:
            cv2.imwrite(output_path, vis_img)
        
        if debug_dir:
            cv2.imwrite(f"{debug_dir}/8_final_result.png", vis_img)
        
        return solution_text


