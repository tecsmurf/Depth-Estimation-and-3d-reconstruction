## 📌 Depth Estimation and 3D Reconstruction

A deep learning-based system that estimates monocular depth and reconstructs 3D structure from 2D images using fine-tuned models and state-of-the-art vision techniques.

---

### 🧠 Features

- Monocular depth estimation from a single image
- 3D reconstruction using depth maps
- Fine-tuned transformer-based model (`.safetensors`)
- Visual outputs: disparity maps, 3D point clouds, etc.
- Modular code structure for easy experimentation

---

### 📁 Project Structure

```
depth-estimation-3d/
├── fine_tuned_dpt/
│   └── model.safetensors     # Trained model (LFS)
├── utils/
│   └── helpers.py            # Utility functions
├── notebooks/
│   └── inference.ipynb       # Demo notebook
├── requirements.txt
├── main.py                   # CLI interface or main script
├── README.md
└── .gitattributes            # Git LFS tracking
```

---

### 🚀 Getting Started

#### 1. Clone the repository

```bash
git clone https://github.com/your-username/depth-estimation-3d.git
cd depth-estimation-3d
```

#### 2. Set up a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run inference

```bash
python main.py --image input.jpg --output output/
```

Or use the notebook:

```bash
jupyter notebook notebooks/inference.ipynb
```

---

### 📦 Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Matplotlib
- Transformers
- (See `requirements.txt` for details)

---

### 🧠 Model Info

The project uses a fine-tuned [DPT](https://github.com/isl-org/DPT) (Dense Prediction Transformer) model for depth estimation.

> 📁 **Note**: Model weights (`model.safetensors`) are tracked using [Git LFS](https://git-lfs.github.com/)

---

### 📸 Example Output

| Input Image | Depth Map | 3D Reconstruction |
|-------------|-----------|-------------------|
| ![](sample/input.jpg) | ![](sample/depth.jpg) | ![](sample/3d.jpg) |

---

### 📌 TODO

- [x] Add Git LFS support
- [ ] Upload training notebook
- [ ] Integrate 3D mesh exporter
- [ ] Deploy as web app (Streamlit or Gradio)

---

### 🙌 Acknowledgements

- [DPT by Intel ISL](https://github.com/isl-org/DPT)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- Contributors & community



