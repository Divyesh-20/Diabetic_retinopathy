<!-- ======================= HEADER ======================= -->

<h1 align="center">👁️ AI-Powered Diabetic Retinopathy Detection System</h1>

<p align="center">
  <b>Early Detection of Blindness using Deep Learning + Explainable AI</b>
</p>

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?color=36BCF7&lines=Detect+Diabetic+Retinopathy;Explain+AI+Decisions+with+Grad-CAM;Generate+Medical+Reports;Full-Stack+AI+System&center=true&width=500&height=45">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg">
  <img src="https://img.shields.io/badge/Framework-TensorFlow-orange.svg">
  <img src="https://img.shields.io/badge/UI-Streamlit-red.svg">
  <img src="https://img.shields.io/badge/Status-Active-success.svg">
  <img src="https://img.shields.io/badge/License-Research-lightgrey.svg">
</p>

---

## 🌍 Why This Project Matters

- 📈 **100M+ diabetics in India**
- 👁️ Leading cause of **preventable blindness**
- ❌ Problem: Late diagnosis + lack of specialists  

💡 **Solution:** AI-based early screening system for faster detection

---

## 🎯 What This System Does

✔️ Upload retinal image  
✔️ Detect DR stage (0–4)  
✔️ Show confidence score  
✔️ Highlight affected areas (Grad-CAM)  
✔️ Generate PDF medical report  

---

## 🧠 Model Architectures

| Model | Purpose | Strength |
|------|--------|---------|
| CNN + LSTM | Baseline | Stable |
| MobileNetV2 + LSTM | Lightweight | Fast |
| InceptionResNetV2 + LSTM | Primary | High Accuracy |

---

## 🖥️ System Flow

```mermaid
graph LR
A[Upload Image] --> B[Preprocessing CLAHE]
B --> C[Deep Learning Model]
C --> D[Prediction]
D --> E[Grad-CAM Heatmap]
E --> F[PDF Report]
