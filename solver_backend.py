import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sympy

# --- CONFIGURATION ---
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
        try:
            self.model = load_model(model_path)
            self.model_loaded = True
        except:
            self.model_loaded = False

    def pad_to_square(self, img, padding=4):
        """Pads crop to square to preserve aspect ratio (CRITICAL for digit recognition)"""
        h, w = img.shape
        diff = abs(h - w)
        pad1, pad2 = diff // 2, diff - (diff // 2)
        
        if h > w:
            padded = cv2.copyMakeBorder(img, 0, 0, pad1 + padding, pad2 + padding, cv2.BORDER_CONSTANT, value=0)
        else:
            padded = cv2.copyMakeBorder(img, pad1 + padding, pad2 + padding, 0, 0, cv2.BORDER_CONSTANT, value=0)
            
        return padded

    def preprocess_image(self, img):
        """
        Advanced preprocessing to handle shadows and noise.
        Returns a clean binary image (white text on black background).
        """
        # 1. Grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 2. Gaussian Blur (Reduces high-frequency noise/grain)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Adaptive Thresholding (Handles shadows/uneven lighting)
        # Block size 11, C=2. This is the "magic" step for paper.
        # cv2.THRESH_BINARY_INV means: Ink becomes White (255), Paper becomes Black (0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # 4. Morphological Cleanup (Optional but good)
        # Remove tiny noise dots (opening) and close small gaps in ink (closing)
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) # Remove noise
        # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) # Connect broken lines
        
        return binary

    def solve_image(self, image_buffer):
        if not self.model_loaded:
            return None, "Model not found", "N/A"

        # Read Streamlit buffer to OpenCV
        file_bytes = np.asarray(bytearray(image_buffer.read()), dtype=np.uint8)
        original = cv2.imdecode(file_bytes, 1)
        
        # --- NEW PREPROCESSING ---
        thresh = self.preprocess_image(original)
        
        # Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter noise contours (too small)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
        
        boxes = [cv2.boundingRect(c) for c in valid_contours]
        boxes = sorted(boxes, key=lambda b: b[0]) # Sort Left to Right
        
        # Merge Overlaps (Fixes =, i, %)
        merged_boxes = []
        skip_indices = set()
        
        for i in range(len(boxes)):
            if i in skip_indices: continue
            x1, y1, w1, h1 = boxes[i]
            merged = False
            
            for j in range(i + 1, len(boxes)):
                if j in skip_indices: continue
                x2, y2, w2, h2 = boxes[j]
                
                # Check horizontal distance & vertical alignment
                x_dist = x2 - (x1 + w1)
                x_overlap = max(0, min(x1+w1, x2+w2) - max(x1, x2))
                
                # If they overlap in X or are very close (like : in division)
                if x_overlap > 0 or abs(x_dist) < 15:
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_w = max(x1+w1, x2+w2) - new_x
                    new_h = max(y1+h1, y2+h2) - new_y
                    
                    merged_boxes.append((new_x, new_y, new_w, new_h))
                    skip_indices.add(j)
                    merged = True
                    break
            
            if not merged:
                merged_boxes.append((x1, y1, w1, h1))

        # Prediction Loop
        batch_images = []
        final_boxes = []
        
        for (x, y, w, h) in merged_boxes:
            # Aspect Ratio Filter (Reject long horizontal lines that are noise)
            if w > 3*h and h < 10: continue 
            
            crop = thresh[y:y+h, x:x+w]
            
            # Pad to square -> Resize to 28x28 -> Normalize
            square = self.pad_to_square(crop, padding=6) # Add padding to be safe
            final = cv2.resize(square, (28, 28))
            
            # Dilate slightly to make stroke thicker (matches MNIST style)
            # final = cv2.dilate(final, np.ones((2,2), np.uint8), iterations=1)
            
            final_norm = final.astype('float32') / 255.0
            batch_images.append(final_norm)
            final_boxes.append((x, y, w, h))

        if not batch_images:
            return original, "No symbols detected", "Error"

        # Predict
        batch_input = np.array(batch_images).reshape(-1, 28, 28, 1)
        probs = self.model.predict(batch_input, verbose=0)
        preds = np.argmax(probs, axis=1)
        labels = [ID_TO_LABEL[p] for p in preds]

        # Parse
        equation_str = ""
        for l in labels:
            if l == 'times' or l == 'x' or l == 'X': equation_str += '*'
            elif l == 'div': equation_str += '/'
            elif l in ['plus', 'add']: equation_str += '+'
            elif l in ['minus', 'sub']: equation_str += '-'
            elif l in ['=', 'eq', 'equal']: continue
            else: equation_str += l

        # Solve
        try:
            result = sympy.sympify(equation_str)
            if hasattr(result, 'is_integer') and result.is_integer:
                result = int(result)
        except:
            result = "Error"

        # Draw Visualization
        display_img = original.copy()
        for i, (x, y, w, h) in enumerate(final_boxes):
            # Draw Box
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 200, 0), 2)
            # Draw Label
            cv2.putText(display_img, labels[i], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return display_img, equation_str, result
