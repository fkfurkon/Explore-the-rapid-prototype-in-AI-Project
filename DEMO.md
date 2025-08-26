# 🎯 Demo และตัวอย่างการใช้งาน

## ตัวอย่าง URL ภาพสำหรับทดสอบ

### ภาพธรรมชาติ
- `https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800`
- `https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=800`

### ภาพสถาปัตยกรรม  
- `https://images.unsplash.com/photo-1480714378408-67cf0d13bc1f?w=800`
- `https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800`

### ภาพสัตว์
- `https://images.unsplash.com/photo-1574158622682-e40e69881006?w=800`
- `https://images.unsplash.com/photo-1425082661705-1834bfd09dca?w=800`

### ภาพคน
- `https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800`
- `https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=800`

## วิธีทดสอบฟีเจอร์ต่างๆ

### 1. Camera Input
- เลือก "Camera" จาก dropdown
- คลิก "Take a picture"  
- อนุญาตการเข้าถึงกล้องของเบราว์เซอร์
- ถ่ายภาพ

### 2. URL Input
- เลือก "URL" จาก dropdown
- คัดลอก URL ข้างบนมาใส่
- ภาพจะโหลดอัตโนมัติ

### 3. Upload Image
- เลือก "Upload Image" จาก dropdown
- คลิก "Browse files"
- เลือกไฟล์ภาพจากคอมพิวเตอร์

### 4. Image Processing Parameters

#### Brightness (-100 ถึง 100)
- ค่าบวก = ภาพสว่างขึ้น
- ค่าลบ = ภาพมืดลง

#### Contrast (0.1 ถึง 3.0)
- ค่ามากกว่า 1 = เพิ่มความคมชัด
- ค่าน้อยกว่า 1 = ลดความคมชัด

#### Filter Types
1. **None**: ไม่ใช้ฟิลเตอร์
2. **Gaussian Blur**: เบลอแบบนุ่มนวล
3. **Median Blur**: เบลอที่ช่วยลด noise
4. **Edge Detection**: ตรวจจับขอบวัตถุ

#### Blur Kernel Size (1-15, เลขคี่)
- ยิ่งใหญ่ = เบลอมากขึ้น

#### Edge Detection Thresholds
- **Threshold 1**: ขอบอ่อน
- **Threshold 2**: ขอบแข็ง

## การแปลผลกราฟ

### Color Distribution Histogram
- แสดงการกระจายของความเข้มสีในแต่ละ channel (R, G, B)
- Peak สูง = มีพิกเซลจำนวนมากที่ความเข้มนั้น
- กราฟแบน = สีกระจายทั่วทั้งภาพ

### Mean Intensity Comparison
- เปรียบเทียบค่าเฉลี่ยของสีระหว่างภาพต้นฉบับกับภาพที่ประมวลผล
- ช่วยดูผลกระทบของการปรับแต่ง

## เทคนิคการใช้งานขั้นสูง

### สำหรับภาพที่มี Noise มาก
1. ใช้ Median Blur ก่อน
2. ปรับ Contrast เล็กน้อย
3. ใช้ Edge Detection หากต้องการ

### สำหรับภาพที่มืด
1. เพิ่ม Brightness (20-50)
2. เพิ่ม Contrast (1.2-1.5)

### สำหรับภาพที่เบลอ
1. เพิ่ม Contrast (1.5-2.0)
2. อาจใช้ Edge Detection เพื่อเน้นขอบ

## การ Export และ Save
- คลิก "Download Processed Image" เพื่อบันทึกภาพ
- ภาพจะบันทึกในรูปแบบ PNG
- มีชื่อไฟล์ "processed_image.png"
