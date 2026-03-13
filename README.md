# Diabetic Retinopathy Detection System 👁️🩺

A full-stack, end-to-end Deep Learning platform for detecting and grading Diabetic Retinopathy (DR) from retinal fundus images. Built natively in Python with a comprehensive **Streamlit** dashboard, this project features 9 robust CNN and hybrid LSTM architectures, clinical explainability via Grad-CAM, and dynamically generated medical PDF reports.

---

## 🚀 Features

- **Dual-Role Dashboard (Admin & User):** Secure session-state role management separates clinical inference from model engineering.
- **9 Interchangeable Deep Learning Models:** Easily train, evaluate, and switch between custom architectures:
  - Custom AlexNet, DenseNet121, InceptionV3, EfficientNetB0, ResNet50, MobileNetV2.
  - Advanced Sequence Hybrids: CNN+LSTM, MobileNetV2+LSTM, and the primary **InceptionResNetV2 + LSTM**.
- **Automated Imbalance Handling:** Dynamic class weighting and stratified splitting handles real-world medical data imbalances gracefully.
- **Explainable AI (XAI):** Integrated **Grad-CAM** generates vibrant heatmap overlays, highlighting the exact retinal regions triggering the model's diagnosis.
- **Clinical Admin Panel:** Live epoch-by-epoch training charts, detailed confusion matrices, per-class ROC curves, and side-by-side model comparison metrics (Quadratic Kappa, F1-Macro, AUC).
- **Exportable Medical Reports:** Single-click generation of styled, patient-ready PDF diagnostic reports.

---

## 🛠️ Tech Stack

- **Frontend / UI:** [Streamlit](https://streamlit.io/) with custom dark medical CSS aesthetics.
- **Deep Learning Backend:** [TensorFlow 2 / Keras](https://tensorflow.org/)
- **Data Engineering:** OpenCV (CLAHE Enhancement), NumPy, Pandas
- **Visualization:** Matplotlib, Plotly, Seaborn
- **Utilities:** ReportLab (PDF Generation), Scikit-Learn (Metrics)

---

## 📂 Project Structure

```bash
├── app.py                     # Main Streamlit Entry Router
├── config.py                  # Global settings, paths, and model registry
├── requirements.txt           # Python dependencies
├── models/                    # 9 DL architectures + unified Model Factory
├── pages/
│   ├── login.py               # Auth portal
│   ├── admin/                 # Dashboard, Train, Evaluate, Compare pages
│   └── user/                  # Inference, Grad-CAM, and PDF Download
├── utils/                     
│   ├── auth.py                # Session management
│   ├── dataset.py             # Data loader, stratifier, and class weighter
│   ├── gradcam.py             # Heatmap generation
│   ├── metrics.py             # Cohen's Kappa, AUC, F1 calculators
│   ├── preprocessing.py       # CLAHE & image normalization
│   ├── report_generator.py    # PDF Compiler
│   └── trainer.py             # TF Training Loop with Streamlit Callbacks
└── dataset/                   # Your fundus image repository (See Setup)
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/diabetic-retinopathy-system.git
   cd diabetic-retinopathy-system
   ```

2. **Install Dependencies:**
   Requires Python 3.9+ (Tested on 3.12.0)
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset:**
   Place your fundus images in the `dataset/` directory. The system automatically detects root `train` and `validation` splits with named subfolders representing the 5 DR stages:
   ```text
   dataset/
   ├── train/
   │   ├── No_DR/
   │   ├── Mild/
   │   ├── Moderate/
   │   ├── Severe/
   │   └── Proliferate_DR/
   └── validation/
       ├── No_DR/ ...
   ```

---

## 💻 Usage

Start the Streamlit web application:
```bash
streamlit run app.py
```

### 🔑 Default Credentials
- **Admin Access:** `admin` / `admin123`
- **User / Patient Access:** `user` / `user123`

### 🏗️ Admin Workflow
1. Navigate to **Train Model**.
2. Select an architecture (e.g., `inception_resnet_lstm`) and set hyperparameters.
3. Watch the live training graphs.
4. Navigate to **Evaluate Model** to view the confusion matrix, ROC curves, and quantitative metrics for the saved `.h5` weights.

### 🏥 User Workflow
1. Log in as a user and upload a raw fundus `.png` or `.jpg`.
2. The system preprocesses the image (CLAHE), runs inference on the active Main Model, and outputs a color-coded DR stage severity.
3. Review the Grad-CAM heatmap overlay.
4. Click **Download Report** for the offline PDF.

---

## 📈 Evaluation Metrics Focus

Medical datasets require stringent grading. This system natively prioritizes:
- **Quadratic Cohen's Kappa:** The standard metric penalizing predictions that are further away from the true severity stage (e.g., guessing Stage 0 when true is Stage 4 is heavily penalized).
- **Macro F1-Score:** Ensures model performance is balanced across rare severe stages, rather than inflating accuracy via the majority "No DR" class.
- **One-vs-Rest ROC AUC:** Granular insight into binary classification capabilities per stage.

---

## 📜 License
This project is for educational and research purposes. Medical decisions should not be based solely on automated software outputs.
