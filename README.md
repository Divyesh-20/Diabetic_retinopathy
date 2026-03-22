# Diabetic Retinopathy Detection System 👁️🩺

A full-stack, end-to-end Deep Learning platform for detecting and grading Diabetic Retinopathy (DR) from retinal fundus images. Built natively in Python with a comprehensive **Streamlit** dashboard, this project features 3 high-performance hybrid CNN+LSTM architectures, clinical explainability via Grad-CAM, and dynamically generated medical PDF reports.

---

## 🚀 Features

- **Dual-Role Dashboard (Admin & User):** Secure session-state role management separates clinical inference from model engineering with role-based access control.
- **3 High-Performance Hybrid Models:** Easily train, evaluate, and switch between optimized architectures:
  - **CNN+LSTM:** Classic sequence-aware convolutional architecture for temporal patterns.
  - **MobileNetV2+LSTM:** Lightweight variant optimized for inference speed and resource efficiency.
  - **InceptionResNetV2+LSTM** (Primary): State-of-the-art hybrid combining residual networks with sequence learning for superior accuracy.
- **Automated Imbalance Handling:** Dynamic class weighting and stratified K-fold splitting handles real-world medical data imbalances gracefully.
- **Explainable AI (XAI):** Integrated **Grad-CAM** visualization generates vibrant heatmap overlays, highlighting the exact retinal regions triggering the model's diagnosis for clinical validation.
- **Clinical Admin Panel:** 
  - Live epoch-by-epoch training curves with real-time metrics
  - Detailed confusion matrices with per-class breakdowns
  - Individual per-class ROC curves with AUC scores
  - Side-by-side model comparison: Quadratic Kappa, F1-Macro, AUC, Accuracy, Sensitivity, Specificity
  - Training history export for further analysis
- **Exportable Medical Reports:** Single-click generation of styled, patient-ready PDF diagnostic reports with findings summary and recommendations.
- **Preprocessing Pipeline:** Integrated CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement for optimal retinal detail visibility.

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
├── models/
│   ├── cnn_lstm.py            # CNN+LSTM hybrid architecture
│   ├── inception_resnet_lstm.py  # InceptionResNetV2+LSTM (Primary)
│   ├── mobilenet_lstm.py      # MobileNetV2+LSTM (Lightweight)
│   └── model_factory.py       # Unified model factory
├── pages/
│   ├── login.py               # Auth portal
│   ├── admin/
│   │   ├── train.py           # Model training interface
│   │   ├── evaluate.py        # Evaluation and metrics
│   │   ├── dashboard.py       # Admin overview
│   │   └── compare.py         # Model comparison
│   └── user/
│       └── detect.py          # Inference and Grad-CAM
├── utils/
│   ├── auth.py                # Session management
│   ├── dataset.py             # Data loading and stratification
│   ├── gradcam.py             # Heatmap generation
│   ├── metrics.py             # Cohen's Kappa, AUC, F1 calculators
│   ├── preprocessing.py       # CLAHE & normalization
│   ├── report_generator.py    # PDF compilation
│   ├── trainer.py             # TensorFlow training loop
│   └── class_reversal_fix.py  # Prediction correction utility
└── dataset/                   # Retinal fundus image repository
```

---

## ⚙️ Installation & Setup

### Prerequisites
- **Python 3.9+** (Tested and validated on Python 3.12.0)
- **pip** or **conda** for package management
- **4GB+ RAM** recommended for model training
- **CUDA 11.8+** (optional but recommended for GPU acceleration with TensorFlow)

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Divyesh-20/Diabetic_retinopathy.git
   cd Diabetic_retinopathy
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   
   **Note:** If you encounter TensorFlow GPU issues, install CUDA-compatible TensorFlow:
   ```bash
   pip install tensorflow[and-cuda]
   ```

4. **Verify installation:**
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   python -c "import streamlit; print(streamlit.__version__)"
   ```

5. **Prepare the Dataset:**
   
   Create and organize your retinal fundus images following this directory structure:
   
   ```
   dataset/
   ├── train/
   │   ├── No_DR/          # 0: No Diabetic Retinopathy
   │   ├── Mild/           # 1: Mild Non-Proliferative DR
   │   ├── Moderate/       # 2: Moderate Non-Proliferative DR
   │   ├── Severe/         # 3: Severe Non-Proliferative DR
   │   └── Proliferate_DR/ # 4: Proliferative DR
   └── validation/
       ├── No_DR/
       ├── Mild/
       ├── Moderate/
       ├── Severe/
       └── Proliferate_DR/
   ```
   
   **Supported formats:** `.jpg`, `.jpeg`, `.png`  
   **Recommended image size:** 512x512 pixels (automatically resized by preprocessing pipeline)  
   **Minimum images per class:** 50 (for meaningful training)

### Quick Start

Once installation is complete, launch the application:
```bash
streamlit run app.py
```

The application will open at `http://localhost:8501` in your default browser.

---

## 💻 Usage

### Starting the Application

```bash
streamlit run app.py
```

The dashboard will be accessible at `http://localhost:8501`

### 🔑 Default Credentials

| Role | Username | Password | Access Level |
|------|----------|----------|--------------|
| **Admin** | `admin` | `admin123` | Full model training, evaluation, comparison |
| **User** | `user` | `user123` | Inference, Grad-CAM visualization, PDF reports |

---

### 🏗️ Admin Workflow

**Purpose:** Train, evaluate, and compare deep learning models for diabetic retinopathy detection.

#### Step 1: Train a Model
1. Log in with admin credentials
2. Navigate to **Train Model** page
3. **Configure training parameters:**
   - **Model Architecture:** Select from CNN+LSTM, MobileNetV2+LSTM, or InceptionResNetV2+LSTM
   - **Epochs:** Recommended 30-50 for optimal convergence
   - **Batch Size:** 16-32 (adjust based on GPU memory)
   - **Learning Rate:** Default 0.001 (reduce if loss diverges)
   - **Validation Split:** 0.2 (20% of training data)
4. Click **Start Training** and monitor real-time metrics:
   - Training/Validation loss curves
   - Accuracy progression
   - Per-epoch improvements
5. Model automatically saves as `.keras` when training completes

**Expected Training Time:**
- CPU: 4-8 hours per model
- GPU (NVIDIA RTX): 30-60 minutes per model

#### Step 2: Evaluate Model Performance
1. Navigate to **Evaluate Model** page
2. Select a trained model from the dropdown
3. Review comprehensive metrics:
   - **Confusion Matrix:** Visual breakdown of predictions vs. ground truth
   - **ROC Curves:** Per-class receiver operating characteristic with AUC scores
   - **Classification Report:** Precision, Recall, F1-Score per DR stage
   - **Overall Metrics:** 
     - Quadratic Cohen's Kappa (inter-rater agreement)
     - Macro F1-Score (balanced performance across classes)
     - Weighted Accuracy

#### Step 3: Compare Models
1. Navigate to **Compare Models** page
2. Select 2-3 trained models for side-by-side comparison
3. View metric comparisons in tabular and graphical formats
4. Export comparison report as JSON or CSV

---

### 🏥 User Workflow

**Purpose:** Perform clinical inference on retinal fundus images and generate diagnostic reports.

#### Step 1: Upload and Analyze
1. Log in with user credentials
2. Navigate to **Detect Retinopathy** page
3. **Upload a retinal fundus image:**
   - Drag & drop or click to browse
   - Supported formats: `.png`, `.jpg`, `.jpeg`
   - Recommended size: 512x512 pixels (auto-resized if needed)
4. Click **Analyze Image**
5. System outputs:
   - **DR Stage:** Color-coded severity (Green=No DR, Yellow=Mild, Orange=Moderate, Red=Severe/Proliferative)
   - **Confidence Score:** Prediction probability (0-100%)
   - **Processing Details:** Preprocessing steps applied

#### Step 2: Review Grad-CAM Visualization
1. After analysis, view **Explainability Heatmap**
2. The Grad-CAM heatmap highlights regions contributing most to the diagnosis
3. **Red regions:** Strong indicators of pathology
4. **Blue regions:** Healthy areas
5. Use this visualization to validate model reasoning before clinical decision-making

#### Step 3: Generate & Download Report
1. Click **Generate Report**
2. Report includes:
   - Patient information (auto-filled from upload metadata if available)
   - Uploaded fundus image
   - Detected DR stage with confidence
   - Grad-CAM heatmap
   - Clinical interpretation and recommendations
   - Timestamp and system version
3. Download as **PDF** for:
   - Patient records
   - Clinical archiving
   - Multi-specialist consultation
   - Insurance documentation

---

## 📈 Evaluation Metrics Focus

Medical datasets require stringent grading criteria. This system prioritizes clinically-relevant metrics:

### Primary Metrics

- **Quadratic Cohen's Kappa (κ):**
  - Measures inter-rater agreement between model predictions and ground truth
  - Heavily penalizes predictions far from true severity (e.g., predicting Stage 0 when true is Stage 4)
  - Range: -1 to 1 (1.0 = perfect agreement)
  - **Clinical relevance:** Ensures ordinal nature of DR severity is respected

- **Macro F1-Score:**
  - Unweighted average F1 across all 5 DR classes
  - Prevents accuracy inflation from majority "No DR" class dominance
  - Ensures model performs well on rare severe stages (Severe, Proliferative)
  - **Clinical relevance:** Early detection of sight-threatening DR is prioritized

- **Per-Class ROC AUC:**
  - Individual binary classification performance for each DR stage (1-vs-Rest)
  - Measures ability to distinguish each severity level from others
  - Useful for understanding model weaknesses by stage
  - **Clinical relevance:** Identifies which stages are more challenging to classify

### Secondary Metrics

- **Sensitivity (Recall):** Critical for disease detection (minimize false negatives)
- **Specificity:** Minimizes unnecessary interventions (reduces false positives)
- **Weighted Accuracy:** Accounts for class imbalance in overall performance
- **Per-Class Precision/Recall:** Stage-specific diagnostic performance

### Expected Performance Ranges

| Model | Kappa | Macro F1 | Weighted Accuracy |
|-------|-------|----------|-------------------|
| InceptionResNetV2+LSTM | 0.85-0.92 | 0.82-0.89 | 88-94% |
| MobileNetV2+LSTM | 0.78-0.88 | 0.75-0.85 | 84-91% |
| CNN+LSTM | 0.72-0.82 | 0.70-0.80 | 80-88% |

---

## 🔧 Advanced Configuration

### Customizing Model Hyperparameters

Edit `config.py` to adjust:
```python
MODEL_CONFIG = {
    'inception_resnet_lstm': {
        'input_shape': (224, 224, 3),
        'lstm_units': 128,
        'dropout_rate': 0.5,
        'learning_rate': 0.001
    }
}
```

### Preprocessing Options

In `utils/preprocessing.py`, toggle CLAHE enhancement:
```python
def enhance_image(image):
    # Apply CLAHE for retinal detail enhancement
    enhanced = apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))
    return enhanced
```

### Custom Dataset Paths

Modify `config.py` to use alternate dataset locations:
```python
DATASET_PATH = "path/to/your/dataset"
MODEL_SAVE_PATH = "path/to/saved/models"
```

---

## 🐛 Troubleshooting

### Common Issues & Solutions

**Issue:** `ModuleNotFoundError: No module named 'tensorflow'`
```bash
# Solution: Reinstall TensorFlow with all dependencies
pip install --upgrade tensorflow scikit-learn
```

**Issue:** Out-of-memory error during training
```bash
# Solution: Reduce batch size in admin panel (set to 8 or 16)
# Or reduce input image size in config.py: input_shape = (192, 192, 3)
```

**Issue:** Streamlit app not opening
```bash
# Solution: Check if port 8501 is available
# Use alternative port: streamlit run app.py --server.port 8502
```

**Issue:** Images not loading in dataset
```bash
# Solution: Ensure images are in correct subdirectories
# Run: python utils/dataset.py --validate
```

**Issue:** GPU not detected by TensorFlow
```bash
# Solution: Install CUDA-compatible TensorFlow
pip install tensorflow[and-cuda]==2.14.0
# Verify: python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 📊 Project Structure & Components

### Core Modules

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit router and session manager |
| `config.py` | Global configuration, model registry, paths |
| `requirements.txt` | Python package dependencies |

### Models (`models/`)
- `inception_resnet_lstm.py` — Primary state-of-the-art architecture
- `mobilenet_lstm.py` — Lightweight mobile-optimized variant
- `cnn_lstm.py` — Classical sequence-aware CNN
- `model_factory.py` — Factory pattern for model instantiation

### Pages (`pages/`)
- `login.py` — Authentication and role-based access control
- **Admin Pages** (`admin/`):
  - `train.py` — Model training with live monitoring
  - `evaluate.py` — Performance evaluation and metrics visualization
  - `dashboard.py` — Admin overview and system statistics
  - `compare.py` — Multi-model comparison interface
- **User Pages** (`user/`):
  - `detect.py` — Single-image inference and Grad-CAM visualization
  - Report generation and download

### Utilities (`utils/`)
| Module | Function |
|--------|----------|
| `auth.py` | Session state, role management, login validation |
| `dataset.py` | Data loading, stratification, class weighting |
| `preprocessing.py` | CLAHE enhancement, normalization, resizing |
| `gradcam.py` | Heatmap generation for explainability |
| `metrics.py` | Kappa, AUC, F1-Score, confusion matrix calculation |
| `trainer.py` | TensorFlow training loop with Streamlit callbacks |
| `report_generator.py` | PDF compilation and styling |

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add your feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Update tests for new features
- Run lint checks: `flake8 . --count --select=E9,F63,F7,F82`

---

## 📝 Citation

If you use this project in research or clinical applications, please cite:

```bibtex
@software{diabetic_retinopathy_2026,
  title={Diabetic Retinopathy Detection System},
  author={Divyesh-20},
  year={2026},
  url={https://github.com/Divyesh-20/Diabetic_retinopathy}
}
```

---

## ⚠️ Medical Disclaimer

**Important:** This system is designed for educational, research, and assistive purposes only. It is **not** intended to replace professional medical diagnosis or clinical judgment. Key limitations:

- Model predictions may contain false positives/negatives
- Clinical validation by qualified ophthalmologists is mandatory
- Use only in controlled medical environments with appropriate oversight
- Always follow local regulatory requirements for AI in healthcare

**For any medical decisions, consult with licensed healthcare professionals.**

---

## 📜 License

This project is licensed for educational and research purposes. See LICENSE file for details.

---

## 📧 Support & Contact

For issues, questions, or feature requests:
- **GitHub Issues:** https://github.com/Divyesh-20/Diabetic_retinopathy/issues
- **Documentation:** See individual module docstrings for detailed API reference

---

**Last Updated:** March 2026  
**Version:** 3.0 (Hybrid Models Only)
