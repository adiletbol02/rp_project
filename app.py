import streamlit as st
import cv2
import numpy as np
from solver_backend import EquationSolver
from PIL import Image

# Page Config
st.set_page_config(page_title="AI Math Solver", page_icon="🧮")

st.title("🧮 Handwritten Math Solver")
st.write("Upload a photo of a math equation (digits 0-9, +, -, *, /) and I will solve it.")

# Initialize Solver (Cached so it doesn't reload every time)
@st.cache_resource
def get_solver():
    return EquationSolver(model_path='math_cnn.h5')

solver = get_solver()

if not solver.model_loaded:
    st.error("⚠️ Error: 'math_cnn.h5' model not found. Please upload the model file to the root directory.")
else:
    # File Uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display Columns
        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        with st.spinner('Analyzing...'):
            # Run the backend logic
            # We reset the file pointer just in case
            uploaded_file.seek(0) 
            processed_img, equation, result = solver.solve_image(uploaded_file)
            
            # Convert BGR (OpenCV) to RGB (Streamlit)
            processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

        with col2:
            st.image(processed_img_rgb, caption="AI Detection", use_column_width=True)

        # Results Section
        st.divider()
        st.subheader("Results")
        
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**Detected Equation:**")
            st.code(f"{equation}")
        with c2:
            if result == "Error":
                st.error(f"**Solution:** {result}")
            else:
                st.success(f"**Solution:** {result}")

        # Latex display (Optional beautiful math rendering)
        try:
            st.latex(f"{equation} = {result}")
        except:
            pass