                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests
from io import BytesIO

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Image Processing App",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📸 Image Processing Application")
st.markdown("---")

# Sidebar สำหรับควบคุมพารามิเตอร์
st.sidebar.title("🎛️ Control Panel")

# เลือกแหล่งที่มาของภาพ
image_source = st.sidebar.selectbox(
    "Select Image Source",
    ["Upload Image", "Camera", "URL"]
)

# ฟังก์ชันสำหรับโหลดภาพจาก URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")
        return None

# ฟังก์ชันสำหรับการประมวลผลภาพ
def apply_image_processing(image, brightness, contrast, blur_kernel, edge_threshold1, edge_threshold2, filter_type):
    # แปลงเป็น numpy array
    img_array = np.array(image)
    
    # แปลงเป็น BGR สำหรับ OpenCV
    if len(img_array.shape) == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    
    # ปรับความสว่างและความคมชัด
    img_processed = cv2.convertScaleAbs(img_bgr, alpha=contrast, beta=brightness)
    
    # ใช้ฟิลเตอร์ตามที่เลือก
    if filter_type == "Gaussian Blur":
        if blur_kernel > 0:
            img_processed = cv2.GaussianBlur(img_processed, (blur_kernel, blur_kernel), 0)
    elif filter_type == "Edge Detection":
        gray = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, edge_threshold1, edge_threshold2)
        img_processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif filter_type == "Median Blur":
        if blur_kernel > 0:
            img_processed = cv2.medianBlur(img_processed, blur_kernel)
    
    # แปลงกลับเป็น RGB สำหรับแสดงผล
    if len(img_processed.shape) == 3:
        img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
    
    return img_processed

# ฟังก์ชันสำหรับคำนวณสถิติของภาพ
def calculate_image_stats(image):
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        # สำหรับภาพสี
        stats = {
            'Red Channel': {
                'Mean': np.mean(img_array[:,:,0]),
                'Std': np.std(img_array[:,:,0]),
                'Min': np.min(img_array[:,:,0]),
                'Max': np.max(img_array[:,:,0])
            },
            'Green Channel': {
                'Mean': np.mean(img_array[:,:,1]),
                'Std': np.std(img_array[:,:,1]),
                'Min': np.min(img_array[:,:,1]),
                'Max': np.max(img_array[:,:,1])
            },
            'Blue Channel': {
                'Mean': np.mean(img_array[:,:,2]),
                'Std': np.std(img_array[:,:,2]),
                'Min': np.min(img_array[:,:,2]),
                'Max': np.max(img_array[:,:,2])
            }
        }
    else:
        # สำหรับภาพเทา
        stats = {
            'Grayscale': {
                'Mean': np.mean(img_array),
                'Std': np.std(img_array),
                'Min': np.min(img_array),
                'Max': np.max(img_array)
            }
        }
    
    return stats

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📥 Input Image")
    
    # โหลดภาพตามแหล่งที่เลือก
    image = None
    
    if image_source == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
    elif image_source == "Camera":
        st.info("📱 Click the button below to capture from camera")
        picture = st.camera_input("Take a picture")
        if picture is not None:
            image = Image.open(picture)
            
    elif image_source == "URL":
        url = st.text_input("Enter image URL:", placeholder="https://example.com/image.jpg")
        if url:
            image = load_image_from_url(url)
    
    if image is not None:
        st.image(image, caption="Original Image", use_container_width=True)

# Sidebar controls สำหรับ image processing
if image is not None:
    st.sidebar.markdown("### 🎨 Image Processing Controls")
    
    # ความสว่าง
    brightness = st.sidebar.slider("Brightness", -100, 100, 0)
    
    # ความคมชัด
    contrast = st.sidebar.slider("Contrast", 0.1, 3.0, 1.0, 0.1)
    
    # ประเภทฟิลเตอร์
    filter_type = st.sidebar.selectbox(
        "Filter Type",
        ["None", "Gaussian Blur", "Median Blur", "Edge Detection"]
    )
    
    # Blur kernel size (ต้องเป็นเลขคี่)
    blur_kernel = st.sidebar.slider("Blur Kernel Size", 1, 15, 1, 2)
    
    # Edge detection thresholds
    if filter_type == "Edge Detection":
        edge_threshold1 = st.sidebar.slider("Edge Threshold 1", 0, 255, 100)
        edge_threshold2 = st.sidebar.slider("Edge Threshold 2", 0, 255, 200)
    else:
        edge_threshold1, edge_threshold2 = 100, 200

with col2:
    st.header("📤 Processed Image")
    
    if image is not None:
        # ประมวลผลภาพ
        processed_image = apply_image_processing(
            image, brightness, contrast, blur_kernel, 
            edge_threshold1, edge_threshold2, filter_type
        )
        
        # แสดงภาพที่ประมวลผลแล้ว
        st.image(processed_image, caption="Processed Image", use_container_width=True)
        
        # ปุ่มสำหรับดาวน์โหลดภาพ
        processed_pil = Image.fromarray(processed_image)
        buf = BytesIO()
        processed_pil.save(buf, format='PNG')
        byte_im = buf.getvalue()
        
        st.download_button(
            label="📥 Download Processed Image",
            data=byte_im,
            file_name="processed_image.png",
            mime="image/png"
        )

# แสดงกราฟสถิติของภาพ
if image is not None:
    st.markdown("---")
    st.header("📊 Image Statistics & Analysis")
    
    # คำนวณสถิติ
    original_stats = calculate_image_stats(image)
    processed_stats = calculate_image_stats(Image.fromarray(processed_image))
    
    # สร้างกราฟแสดงความหนาแน่นของสี
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🎨 Original Image Color Distribution")
        
        # สร้าง histogram
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            fig = go.Figure()
            colors = ['red', 'green', 'blue']
            channels = ['Red', 'Green', 'Blue']
            
            for i, (color, channel) in enumerate(zip(colors, channels)):
                hist, bins = np.histogram(img_array[:,:,i].flatten(), bins=50, range=[0, 255])
                fig.add_trace(go.Scatter(
                    x=bins[:-1], y=hist, 
                    name=channel, 
                    line=dict(color=color),
                    fill='tonexty' if i > 0 else 'tozeroy'
                ))
        else:
            hist, bins = np.histogram(img_array.flatten(), bins=50, range=[0, 255])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=bins[:-1], y=hist, 
                name='Intensity', 
                line=dict(color='gray'),
                fill='tozeroy'
            ))
        
        fig.update_layout(
            title="Color Intensity Distribution",
            xaxis_title="Intensity Value",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.subheader("📈 Image Properties Comparison")
        
        # สร้างกราฟเปรียบเทียบค่าเฉลี่ยของแต่ละ channel
        if len(np.array(image).shape) == 3:
            channels = ['Red', 'Green', 'Blue']
            original_means = [original_stats[f'{ch} Channel']['Mean'] for ch in channels]
            processed_means = [processed_stats[f'{ch} Channel']['Mean'] for ch in channels]
            
            df = pd.DataFrame({
                'Channel': channels + channels,
                'Mean Intensity': original_means + processed_means,
                'Image Type': ['Original'] * 3 + ['Processed'] * 3
            })
            
            fig = px.bar(df, x='Channel', y='Mean Intensity', color='Image Type',
                        title='Mean Intensity Comparison',
                        barmode='group')
        else:
            df = pd.DataFrame({
                'Image Type': ['Original', 'Processed'],
                'Mean Intensity': [original_stats['Grayscale']['Mean'], 
                                 processed_stats['Grayscale']['Mean']]
            })
            
            fig = px.bar(df, x='Image Type', y='Mean Intensity',
                        title='Mean Intensity Comparison')
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # ตารางสถิติ
    st.subheader("📋 Detailed Statistics")
    col5, col6 = st.columns(2)
    
    with col5:
        st.write("**Original Image:**")
        st.json(original_stats)
    
    with col6:
        st.write("**Processed Image:**")
        st.json(processed_stats)

# Footer
st.markdown("---")
st.markdown("### 🛠️ About This App")
st.markdown("""
This image processing application allows you to:
- 📷 Capture images from camera, upload files, or load from URL
- 🎨 Apply various image processing filters
- 📊 Analyze image statistics and color distribution
- 📥 Download processed images

Built with **Streamlit** and **OpenCV**
""")
