import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
from PIL import Image
from utils.preprocessing import preprocess_image
import ui_components
import gradcam

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Egg Analyzer Pro",
    page_icon="🥚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Inject Global CSS
ui_components.inject_custom_css()

# Check authentication
if not st.session_state.authenticated:
    ui_components.render_login()
    st.stop()

# --- CACHED MODEL LOADING ---
@st.cache_resource
def load_models():
    """Load both detection and classification models."""
    try:
        detector = tf.keras.models.load_model('egg_detector_final.h5')
        classifier = tf.keras.models.load_model('fertility_classifier_final.h5')
        
        with open('label_mapping.json', 'r') as f:
            mapping = json.load(f)
        # Reverse mapping: {index: class_name}
        reverse_mapping = {int(v): k for k, v in mapping.items()}
        
        return detector, classifier, reverse_mapping
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def main():
    ui_components.render_sidebar_controls()
    
    # --- HERO HEADER ---
    ui_components.render_hero()
    
    # Load Models
    with st.spinner("🔄 Initializing AI Engine..."):
        detector, classifier, label_map = load_models()
    
    if not detector or not classifier:
        st.stop()

    # --- MAIN LAYOUT ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📸 Image Acquisition")
    uploaded_file = st.file_uploader(
        label="Upload egg image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            # --- MAIN CONTENT GRID ---
            st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)
            
            col_image, col_analysis = st.columns([1, 1.2], gap="large")
            
            # IMAGE COLUMN
            with col_image:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 🖼️ Sample Visual Preview")
                st.image(image, use_container_width=True)
                st.markdown(f"""
                    <div style="text-align: center; margin-top: 1rem;" class="text-muted">
                        Source: {uploaded_file.name} | Dimensions: {image.size[0]}×{image.size[1]}
                    </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # ANALYSIS COLUMN
            with col_analysis:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 🔬 Diagnostic Analysis Module")
                
                if st.button("🚀 INITIATE QUALITY ASSESSMENT", use_container_width=True):
                    # Data holders
                    results = {"is_egg": False, "egg_conf": 0, "cls_name": None, "cls_conf": 0, "cls_pred": None}
                    
                    with st.status("Initializing Deep Learning Framework...", expanded=True) as status:
                        st.write("Analyzing Morphological Features of Egg Sample...")
                        processed_img = preprocess_image(image)
                        
                        st.write("Executing Stage 1: Object Detection...")
                        det_pred = detector.predict(processed_img, verbose=0)
                        is_egg_prob = det_pred[0][0]
                        
                        if det_pred.shape[1] == 1:
                            results["is_egg"] = is_egg_prob > 0.5
                            results["egg_conf"] = is_egg_prob if results["is_egg"] else (1 - is_egg_prob)
                        else:
                            results["is_egg"] = np.argmax(det_pred) == 0
                            results["egg_conf"] = np.max(det_pred)

                        if not results["is_egg"]:
                            status.update(label="Inference Completed: No Biological Sample Detected", state="error", expanded=False)
                        else:
                            st.write("Executing Stage 2: Quality Classification...")
                            cls_pred = classifier.predict(processed_img, verbose=0)[0]
                            results["cls_pred"] = cls_pred
                            results["cls_idx"] = np.argmax(cls_pred)
                            results["cls_name"] = label_map.get(results["cls_idx"], "Unknown").lower()
                            results["cls_conf"] = cls_pred[results["cls_idx"]]
                            status.update(label="Inference Completed Successfully", state="complete", expanded=False)

                    # --- RENDER RESULTS ---
                    if not results["is_egg"]:
                        ui_components.render_status_card("Detection Status", "NO BIOLOGICAL SAMPLE DETECTED", type="danger")
                        ui_components.render_status_card("Index", f"{results['egg_conf']:.1%}", type="danger")
                    else:
                        ui_components.render_status_card("Sample Detection", "VALID SAMPLE DETECTED", type="success")
                        ui_components.render_result_card(results["cls_name"], results["cls_conf"])
                        
                        # --- XAI: Grad-CAM explanation ---
                        with st.expander("🔍 Why did the model say this? (Grad-CAM Explanation)", expanded=False):
                            try:
                                overlay_img, _, explained_idx, explained_conf = gradcam.explain_prediction(
                                    pil_image=image,
                                    model=classifier,
                                    preprocess_fn=preprocess_image,
                                    class_index=int(results["cls_idx"]),
                                )
                                explained_label = label_map.get(explained_idx, "Unknown").capitalize()
                                exp_col1, exp_col2 = st.columns(2)
                                with exp_col1:
                                    st.image(image, caption="Original", use_container_width=True)
                                with exp_col2:
                                    st.image(overlay_img, caption=f"Grad-CAM: regions driving '{explained_label}'", use_container_width=True)
                                st.markdown(
                                    '<p class="text-muted" style="font-size:0.85rem;">'
                                    'Warmer (red/yellow) regions contributed most to this prediction. '
                                    'This is a post-hoc explanation of the model\'s own reasoning, not an '
                                    'independent verification of egg quality.</p>',
                                    unsafe_allow_html=True,
                                )
                            except Exception as exp_err:
                                st.warning(f"Explanation could not be generated: {exp_err}")

                        # Probability breakdown
                        with st.expander("📊 Egg Quality Class Distribution Analysis", expanded=True):
                            cls_pred = results["cls_pred"]
                            top_indices = np.argsort(cls_pred)[::-1]
                            for idx in top_indices:
                                name = label_map.get(idx, f"Class {idx}").capitalize()
                                academic_map = {
                                    "Fertile": "Grade A (Fresh)", 
                                    "Dead": "Grade C (Defective)", 
                                    "Infertile": "Unclassified"
                                }
                                display_name = academic_map.get(name, name)
                                prob = cls_pred[idx]
                                
                                st.markdown(f"""
                                    <div style="margin-bottom: 1rem;">
                                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                            <span style="font-weight: 600;">{display_name}</span>
                                            <span style="font-weight: 700;">{prob:.1%}</span>
                                        </div>
                                        <div style="background: {ui_components.COLORS[st.session_state.theme]['bg']}; height: 8px; border-radius: 4px; overflow: hidden;">
                                            <div style="background: {ui_components.COLORS[st.session_state.theme]['primary']}; width: {prob*100}%; height: 100%;"></div>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Diagnostic Failure: {str(e)}")
    
    else:
        # EMPTY STATE
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
            st.markdown("### ⚡ Performance")
            st.markdown("High-speed inference utilizing optimized convolutional neural networks.")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
            st.markdown("### 🎯 Accuracy")
            st.markdown("Robust quality metrics validated through dual-stage verification.")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
            st.markdown("### 🛡️ Integrity")
            st.markdown("Encrypted access and secure research data handling protocols.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # --- FOOTER ---
    st.markdown(f"""
        <div style="text-align: center; padding: 3rem; opacity: 0.7; font-size: 0.875rem;">
            <p>Developed as part of an academic research project on AI-based poultry quality assessment</p>
            <p style="color: {ui_components.COLORS[st.session_state.theme]['primary']}; font-weight: 600;">
                Nigerian Army University Biu – Artificial Intelligence in Agriculture Research
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()