import streamlit as st
import cv2
import numpy as np
from solver_backend import EquationSolver
from PIL import Image

# Page Config
st.set_page_config(page_title="AI Math Solver", page_icon="🧮")

st.title("🧮 Smart Math Solver v2")
st.write("Upload a photo of a math equation. I can handle basic arithmetic, algebra (2x=8), and messy handwriting.")

# Initialize Solver with the NEW optimized model
@st.cache_resource
def get_solver():
    # Make sure this filename matches your uploaded model
    return EquationSolver(model_path='math_cnn_v2_optimized.h5')

solver = get_solver()

if not solver.model_loaded:
    st.error("⚠️ Error: Model file ('math_cnn_v2_optimized.h5') not found. Please upload it.")
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_file, caption="Original", use_column_width=True)

        with st.spinner('Deskewing & Solving...'):
            uploaded_file.seek(0) 
            # Returns: processed_image (with boxes), equation_string, result_string
            processed_img, equation, result = solver.solve_image(uploaded_file)
            
            # Convert BGR to RGB for Streamlit
            if processed_img is not None:
                processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            else:
                processed_img_rgb = np.zeros((100,100,3), dtype=np.uint8)

        with col2:
            st.image(processed_img_rgb, caption="AI Analysis (Deskewed)", use_column_width=True)

        st.divider()
        st.subheader("Solution")
        
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**Detected:** {equation}")
        with c2:
            if "Error" in str(result):
                st.error(f"**Result:** {result}")
            else:
                st.success(f"**Result:** {result}")
