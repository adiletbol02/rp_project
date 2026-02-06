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
    def __init__(self, model_path='math_cnn_v2_optimized.h5'):
        try:
            self.model = load_model(model_path)
            self.model_loaded = True
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            self.model_loaded = False
            print(f"❌ Failed to load model: {e}")

    def preprocess_image(self, img):
        """
        1. Downscale (for speed/consistency).
        2. Denoise.
        3. Threshold.
        4. Deskew (Straighten).
        """
        # Convert to Grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 1. INTELLIGENT RESIZE
        # Standardize height to 800px to make kernel sizes consistent
        h, w = gray.shape
        target_height = 800
        scale = target_height / h
        new_w = int(w * scale)
        resized = cv2.resize(gray, (new_w, target_height))

        # 2. Denoise (Bilateral is best for preserving edges)
        denoised = cv2.bilateralFilter(resized, 9, 75, 75)

        # 3. Adaptive Thresholding
        binary = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            blockSize=41, 
            C=15
        )

        # 4. Morphological Cleanup (Remove tiny noise)
        kernel = np.ones((2,2), np.uint8)
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 5. Deskew (Straighten the text)
        coords = np.column_stack(np.where(clean > 0))
        if len(coords) == 0: return clean, 1.0 # Empty image
        
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45: angle = -(90 + angle)
        else: angle = -angle
        
        (h, w) = clean.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(clean, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return rotated, scale

    def find_and_filter_contours(self, binary_img):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours]
        
        if not boxes: return []

        # --- MERGE LOGIC (Recursive) ---
        while True:
            merged = False
            new_boxes = []
            skip_indices = set()
            boxes.sort(key=lambda b: b[0]) # Sort by X

            for i in range(len(boxes)):
                if i in skip_indices: continue
                x1, y1, w1, h1 = boxes[i]
                merged_this_iter = False

                for j in range(i + 1, len(boxes)):
                    if j in skip_indices: continue
                    x2, y2, w2, h2 = boxes[j]

                    # Overlap Checks
                    xi1 = max(x1, x2)
                    yi1 = max(y1, y2)
                    xi2 = min(x1 + w1, x2 + w2)
                    yi2 = min(y1 + h1, y2 + h2)
                    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    area1 = w1 * h1
                    area2 = w2 * h2
                    
                    overlap_ratio = inter_area / min(area1, area2) if min(area1, area2) > 0 else 0
                    
                    # Vertical Stacking Check (share X-space, close Y)
                    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    is_stacked = (x_overlap > 0.4 * min(w1, w2)) 
                    
                    if overlap_ratio > 0.5 or is_stacked:
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
            if not merged: break
        
        # --- FIX: SMARTER FILTERING ---
        if not boxes: return []
        areas = [b[2] * b[3] for b in boxes]
        median_area = np.median(areas)
        
        final_boxes = []
        for b in boxes:
            w, h = b[2], b[3]
            area = w * h
            
            # EXCEPTION: Keep it if it looks like a minus sign
            # Condition: Width is at least 2.5x Height (Wide and Short)
            # AND it's not microscopic (at least 1/15th of median)
            is_minus_like = (w > 2.5 * h) and (area > median_area / 15)
            
            # Keep if standard size OR if it's a minus sign
            if (area > median_area / 5) or is_minus_like:
                final_boxes.append(b)
        
        final_boxes.sort(key=lambda b: b[0])
        return final_boxes

    def extract_and_predict(self, binary_img, boxes):
        batch_images = []
        if not boxes: return []

        for (x, y, w, h) in boxes:
            # ROI
            roi = binary_img[y:y+h, x:x+w]
            
            # Pad to Square
            max_dim = max(w, h)
            pad_w = (max_dim - w) // 2
            pad_h = (max_dim - h) // 2
            # Add extra padding (6px) to match MNIST-style training
            roi = cv2.copyMakeBorder(roi, pad_h+4, pad_h+4, pad_w+4, pad_w+4, cv2.BORDER_CONSTANT, value=0)
            
            # Resize
            roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            
            # Normalize
            roi = roi.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=-1)
            batch_images.append(roi)

        batch_input = np.array(batch_images)
        probs = self.model.predict(batch_input, verbose=0)
        preds = np.argmax(probs, axis=1)
        labels = [ID_TO_LABEL[p] for p in preds]
        return labels

    def disambiguate_symbols(self, labels, boxes):
        """Distinguish between 'x' (variable) and 'times' (multiply)"""
        refined = list(labels)
        if len(boxes) < 2: return refined
        
        avg_height = np.median([b[3] for b in boxes])

        for i in range(len(refined)):
            lbl = refined[i]
            if lbl not in ['x', 'times', 'X']: continue
            
            _, y, _, h = boxes[i]
            
            # Logic: Multiplications are usually smaller OR floating
            is_small = h < (0.75 * avg_height)
            
            # Check neighbors for floating status
            neighbor = boxes[i-1] if i > 0 else (boxes[i+1] if i+1 < len(boxes) else None)
            is_floating = False
            if neighbor:
                _, ny, _, nh = neighbor
                if (y + h) < (ny + nh - 0.2 * nh): # Bottom is significantly higher
                    is_floating = True

            if is_small or is_floating:
                refined[i] = 'times'
            else:
                refined[i] = 'x'
        return refined

    def parse_and_solve(self, labels):
        map_op = {
            'times': '*', 'div': '/', 'plus': '+', '-': '-', 'pm': '+', 
            '=': '=', 'x': 'x', 'y': 'y', 'z': 'z', 'pi': 'pi',
            'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'sqrt': 'sqrt'
        }
        
        eq_str = ""
        for lbl in labels:
            eq_str += map_op.get(lbl, lbl)
            
        # Handle trailing "="
        if eq_str.endswith('='): eq_str = eq_str[:-1]
        
        print(f"Solving: {eq_str}")
        
        transformations = (standard_transformations + (implicit_multiplication_application,))
        
        try:
            if '=' not in eq_str:
                expr = parse_expr(eq_str, transformations=transformations)
                res = float(expr)
                if res.is_integer(): res = int(res)
                return eq_str, f"{res}"
            else:
                parts = eq_str.split('=')
                if len(parts) != 2: return eq_str, "Error: Multiple '='"
                lhs_str, rhs_str = parts
                
                if not rhs_str.strip(): # Case "2+2="
                    expr = parse_expr(lhs_str, transformations=transformations)
                    return lhs_str, f"{float(expr)}"

                lhs = parse_expr(lhs_str, transformations=transformations)
                rhs = parse_expr(rhs_str, transformations=transformations)
                
                # Solve for x (or first symbol)
                syms = lhs.free_symbols.union(rhs.free_symbols)
                if not syms:
                    return eq_str, str(sp.simplify(lhs - rhs) == 0)
                
                target = list(syms)[0]
                sol = sp.solve(lhs - rhs, target)
                return eq_str, f"{target} = {sol}"
                
        except Exception as e:
            return eq_str, f"Math Error: {str(e)}"

    def solve_image(self, image_buffer):
        """Main Pipeline"""
        if not self.model_loaded:
            return None, "Model Error", "Check .h5 file"

        # Read Image
        file_bytes = np.asarray(bytearray(image_buffer.read()), dtype=np.uint8)
        original = cv2.imdecode(file_bytes, 1)
        
        # 1. Preprocess (Deskew)
        processed, scale = self.preprocess_image(original)
        
        # 2. Segment
        boxes = self.find_and_filter_contours(processed)
        if not boxes:
            return processed, "No Content", "Empty"
            
        # 3. Predict
        labels = self.extract_and_predict(processed, boxes)
        
        # 4. Disambiguate
        labels = self.disambiguate_symbols(labels, boxes)
        
        # 5. Solve
        final_eq, result = self.parse_and_solve(labels)
        
        # 6. Visualize (Draw on the DESKEWED image)
        vis_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis_img, labels[i], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return vis_img, final_eq, result

