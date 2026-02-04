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
        Aggressive cleanup pipeline for high-res phone photos.
        1. Downscale: Reduces paper grain visibility.
        2. Bilateral Filter: Blurs noise but keeps edges (ink) sharp.
        3. Morphological Open: Deletes tiny specks.
        """
        # 1. Convert to Grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 2. INTELLIGENT RESIZE (The most important step)
        # High-res (4000px) images capture too much texture. 
        # We resize to a fixed height (e.g., 800px) to standardize density.
        h, w = gray.shape
        target_height = 800
        scale = target_height / h
        new_w = int(w * scale)
        resized = cv2.resize(gray, (new_w, target_height))

        # 3. Denoising (Bilateral is better than Gaussian)
        # It smooths flat areas (paper) but preserves edges (ink).
        # d=9, sigmaColor=75, sigmaSpace=75 are standard strong settings.
        denoised = cv2.bilateralFilter(resized, 9, 75, 75)

        # 4. Adaptive Thresholding (Optimized for Paper)
        # blockSize=41 (Large area) -> Ignores small grain, looks at general lighting
        # C=15 (High constant) -> Forces background to be strictly white
        binary = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            blockSize=41, # Large block size ignores texture
            C=15
        )

        # 5. Morphological "Opening" (Erosion -> Dilation)
        # This eats away small "salt" noise (single pixels) then grows the ink back.
        kernel = np.ones((3,3), np.uint8)
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 6. Dilation (Thicken lines)
        # Phone photos often have thin strokes. We thicken them to look like MNIST data.
        thick = cv2.dilate(clean, kernel, iterations=1)

        return thick, scale # Return scale so we can map boxes back to original if needed

    def solve_image(self, image_buffer):
        if not self.model_loaded:
            return None, "Model not found", "N/A"

        file_bytes = np.asarray(bytearray(image_buffer.read()), dtype=np.uint8)
        original = cv2.imdecode(file_bytes, 1)
        
        # --- CALL NEW PREPROCESSOR ---
        thresh, scale = self.preprocess_image(original)
        
        # Find Contours on the PROCESSED (small) image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter Noise: Ignore tiny blobs relative to the new size
        # Area < 50 pixels in an 800px high image is definitely dust
        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
        
        boxes = [cv2.boundingRect(c) for c in valid_contours]
        boxes = sorted(boxes, key=lambda b: b[0])
        
        # ... [Keep your existing "Merge Overlaps" logic here] ...
        # (Copy-paste the merge loop from the previous step)
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
                # Slightly looser check for resized image
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
        # ... [End Merge Logic] ...

        batch_images = []
        final_boxes = [] # Boxes on the resized image
        
        for (x, y, w, h) in merged_boxes:
            # Aspect ratio check
            if w > 4*h: continue 
            
            # Crop from the PROCESSED image (thresh), not original
            crop = thresh[y:y+h, x:x+w]
            
            square = self.pad_to_square(crop, padding=6)
            final = cv2.resize(square, (28, 28))
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

        # Parse & Solve (Keep existing logic)
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
            if hasattr(result, 'is_integer') and result.is_integer: result = int(result)
        except: result = "Error"

        # --- VISUALIZATION ON ORIGINAL IMAGE ---
        # We must map the small boxes back to the huge original image
        display_img = original.copy()
        inverse_scale = 1 / scale
        
        for i, (x, y, w, h) in enumerate(final_boxes):
            # Scale coordinates back up
            orig_x = int(x * inverse_scale)
            orig_y = int(y * inverse_scale)
            orig_w = int(w * inverse_scale)
            orig_h = int(h * inverse_scale)
            
            cv2.rectangle(display_img, (orig_x, orig_y), (orig_x+orig_w, orig_y+orig_h), (0, 200, 0), 5)
            # Make font huge for high-res images
            font_scale = 2.0 if original.shape[0] > 2000 else 1.0
            cv2.putText(display_img, labels[i], (orig_x, orig_y-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 3)

        return display_img, equation_str, result

