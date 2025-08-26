#!/bin/bash

# สคริปต์สำหรับรันแอปพลิเคชัน Image Processing

echo "🚀 Starting Image Processing Application..."
echo "📋 Loading virtual environment..."

# เปิดใช้งาน virtual environment
source .venv/bin/activate

# รันแอปพลิเคชัน Streamlit
echo "🌐 Starting Streamlit server..."
echo "📱 Open your browser and go to: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the application"

streamlit run image_processing_app.py
