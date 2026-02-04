import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sympy

# Map IDs to Labels (Must match your training)
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
    def __init__(self, model_path='math_cnn.h5'):
        # Load model once
        try:
            self.model = load_model(model_path)
            self.model_loaded = True
        except:
            self.model_loaded = False

    def pad_to_square(self, img, padding=4):
        h, w = img.shape
        diff = abs(h - w)
        pad1, pad2 = diff // 2, diff - (diff // 2)
        if h > w:
            padded = cv2.copyMakeBorder(img, 0, 0, pad1 + padding, pad2 + padding, cv2.BORDER_CONSTANT, value=0)
        else:
            padded = cv2.copyMakeBorder(img, pad1 + padding, pad2 + padding, 0, 0, cv2.BORDER_CONSTANT, value=0)
        return padded

    def solve_image(self, image_buffer):
        if not self.model_loaded:
            return None, "Model not found", None

        # Convert uploaded buffer to OpenCV Image
        file_bytes = np.asarray(bytearray(image_buffer.read()), dtype=np.uint8)
        original = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        # 1. Segmentation
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours]
        boxes = sorted(boxes, key=lambda b: b[0])
        
        # Merge Overlaps
        merged_boxes = []
        skip_indices = set()
        for i in range(len(boxes)):
            if i in skip_indices: continue
            x1, y1, w1, h1 = boxes[i]
            merged = False
            for j in range(i + 1, len(boxes)):
                if j in skip_indices: continue
                x2, y2, w2, h2 = boxes[j]
                x_overlap = max(0, min(x1+w1, x2+w2) - max(x1, x2))
                if x_overlap > 0 or abs(x1 - x2) < 10:
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_w = max(x1+w1, x2+w2) - new_x
                    new_h = max(y1+h1, y2+h2) - new_y
                    merged_boxes.append((new_x, new_y, new_w, new_h))
                    skip_indices.add(j)
                    merged = True
                    break
            if not merged: merged_boxes.append((x1, y1, w1, h1))

        # 2. Prediction
        batch_images = []
        valid_boxes = []
        for (x, y, w, h) in merged_boxes:
            if w < 5 or h < 5: continue
            crop = thresh[y:y+h, x:x+w]
            square = self.pad_to_square(crop, padding=4)
            final = cv2.resize(square, (28, 28))
            final_norm = final.astype('float32') / 255.0
            batch_images.append(final_norm)
            valid_boxes.append((x, y, w, h))
            
        if not batch_images:
            return original, "No symbols detected", "N/A"
            
        batch_input = np.array(batch_images).reshape(-1, 28, 28, 1)
        probs = self.model.predict(batch_input, verbose=0)
        preds = np.argmax(probs, axis=1)
        labels = [ID_TO_LABEL[p] for p in preds]

        # 3. Parsing
        equation_str = ""
        for l in labels:
            if l == 'times' or l == 'x' or l == 'X': equation_str += '*'
            elif l == 'div': equation_str += '/'
            elif l in ['plus', 'add']: equation_str += '+'
            elif l in ['minus', 'sub']: equation_str += '-'
            elif l in ['=', 'eq', 'equal']: continue
            else: equation_str += l
            
        try:
            result = sympy.sympify(equation_str)
            if hasattr(result, 'is_integer') and result.is_integer:
                result = int(result)
        except:
            result = "Error"

        # 4. Draw boxes on original image for display
        display_img = original.copy()
        for (x, y, w, h) in valid_boxes:
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return display_img, equation_str, result