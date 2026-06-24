# 🛒 Cart Detector AI

CNN-based image classifier that detects whether a cart/crate is **empty**, **filled**, or **not a cart at all**.

## Use Case
Warehouse automation — check if a cart needs restocking or if it already has items.

## Classes
| Class | Description |
|-------|-------------|
| 🟢 `empty_cart` | Cart with nothing in it |
| 🔴 `filled_cart` | Cart with boxes, items, etc. |
| 🔵 `not_a_cart` | Not a cart (dog, cat, person, etc.) |

## Project Structure
```
cart_detector_project/
├── dataset/
│   ├── train/
│   │   ├── empty_cart/     (33 images)
│   │   ├── filled_cart/    (33 images)
│   │   └── not_a_cart/     (22 images)
│   ├── test/
│   │   ├── empty_cart/
│   │   ├── filled_cart/
│   │   └── not_a_cart/
│   └── validation/
│       ├── empty_cart/     (1_empty_cart.jpg, 4_empty_cart.jpg)
│       ├── filled_cart/    (2_filled_cart.jpg, 5_filled_cart.jpg)
│       └── not_a_cart/     (3_not_cart.jpg, 6_not_cart.jpg)
├── model/
│   └── cart_detector_model.pkl
├── app.py                  (Streamlit UI)
├── train_model.py          (Training script)
└── requirements.txt
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (optional — pre-trained model included)
```bash
python train_model.py
```

### 3. Launch the web app
```bash
streamlit run app.py
```

### 4. Upload an image
- Drag and drop or click to upload
- Get instant prediction with confidence %

## How It Works
1. **Feature Extraction**: Extracts 34 features from each image:
   - Color statistics (mean, std of R/G/B)
   - Edge detection (filled carts have more edges)
   - Blur/texture analysis
   - Brightness distribution
   - Color histograms

2. **Classification**: Random Forest classifier predicts the class

3. **Output**: Shows predicted class + confidence % for all 3 classes

## Dataset
- Training: 88 images (augmented from 8 base images)
- Validation: 6 numbered images
- Images sourced from web + user uploads
