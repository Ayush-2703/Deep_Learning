# 🖼️ Image Processing with Python — Blur & Sharpen Filters

A Google Colab-based image processing project that demonstrates classical spatial filtering techniques using **OpenCV**, **Pillow (PIL)**, **NumPy**, and **Matplotlib**. This project applies and compares Gaussian blur, PIL sharpening, and custom convolution-based sharpening kernels on uploaded images.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Theory of Digital Image Processing](#theory-of-digital-image-processing)
  - [What is Spatial Filtering?](#what-is-spatial-filtering)
  - [Convolution Operation](#convolution-operation)
  - [Gaussian Blur](#gaussian-blur)
  - [Image Sharpening](#image-sharpening)
  - [Custom Sharpening Kernels](#custom-sharpening-kernels)
- [Filters Used in This Project](#filters-used-in-this-project)
  - [Gaussian Blur Filter](#1-gaussian-blur-filter)
  - [PIL Sharpen Filter](#2-pil-sharpen-filter)
  - [Volvo8 Sharpen Filter](#3-volvo8-sharpen-filter)
  - [General Sharpen Filter](#4-general-sharpen-filter)
- [Libraries Used](#libraries-used)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Output Description](#output-description)
- [Applications](#applications)
- [References](#references)

---

## Overview

Image processing is a core area of computer vision that involves modifying or analyzing digital images through mathematical operations. This project focuses on **spatial domain filtering** — one of the most fundamental techniques — to perform:

- **Blurring** — Smooths the image by reducing noise and high-frequency details.
- **Sharpening** — Enhances edges and fine details that may be blurred or unclear.

The project visually compares five versions of the same image side by side so the effect of each filter is clearly observable.

---

## Theory of Digital Image Processing

### What is Spatial Filtering?

A digital image is a 2D array (matrix) of pixels, where each pixel holds an intensity value (0–255 for 8-bit grayscale, or three such values for RGB channels). **Spatial filtering** operates directly on these pixel values using a small matrix called a **kernel** (also known as a filter or mask) that slides over the image.

Each output pixel is computed as a **weighted sum** of the pixel and its neighbors, using the kernel weights.

### Convolution Operation

The mathematical operation behind spatial filtering is called **2D convolution**:

```
Output(x, y) = Σ Σ Kernel(i, j) × Image(x+i, y+j)
               i  j
```

Where:
- `Image(x, y)` is the input pixel at position (x, y)
- `Kernel(i, j)` is the kernel weight at position (i, j)
- `Output(x, y)` is the resulting pixel

The kernel size determines how many neighbors are considered. A **3×3 kernel** (the most common) considers each pixel and its 8 immediate neighbors.

---

### Gaussian Blur

Gaussian blur is the most widely used smoothing filter. It applies a **Gaussian function** (bell curve) to weight neighboring pixels — closer pixels get higher weight, distant ones get lower weight.

**Gaussian function in 2D:**

```
G(x, y) = (1 / 2πσ²) × e^(-(x² + y²) / 2σ²)
```

Where:
- `σ` (sigma) controls the spread — a larger sigma means more blurring
- `x, y` are coordinates relative to the kernel center

**Example Gaussian kernel (3×3 approximation):**

```
| 1  2  1 |
| 2  4  2 | × (1/16)
| 1  2  1 |
```

**Why use it?**
- Removes high-frequency noise (salt-and-pepper noise, random pixel variations)
- Pre-processing step before edge detection (e.g., Canny edge detector)
- Creates a soft, natural-looking blur

The `radius` parameter in `PIL`'s `GaussianBlur` controls how far the effect spreads (equivalent to σ).

---

### Image Sharpening

Sharpening works by **enhancing high-frequency components** of an image — primarily edges and fine details. High-frequency regions are areas where pixel intensity changes rapidly (edges), while low-frequency regions change gradually (flat surfaces).

The idea is:

```
Sharpened Image = Original Image + α × (High-Frequency Detail)
```

High-frequency detail is obtained by subtracting a blurred version from the original:

```
Detail = Original − Blurred
Sharpened = Original + α × Detail
```

This is also known as **Unsharp Masking**, a technique originally from darkroom photography.

---

### Custom Sharpening Kernels

Sharpening can also be implemented directly using a convolution kernel that combines the original and its Laplacian in one step.

**The Laplacian operator** detects areas of rapid intensity change by computing a second-order spatial derivative:

```
∇²f = ∂²f/∂x² + ∂²f/∂y²
```

By subtracting the Laplacian from the original, we get a sharper image:

```
Sharpened = Original − ∇²(Original)
```

The discrete 3×3 Laplacian-based sharpening kernel used in this project is:

```
|  0  -1   0 |
| -1   5  -1 |
|  0  -1   0 |
```

**How this kernel works:**
- The center value `5` amplifies the current pixel
- The `-1` values subtract the contribution of the 4 adjacent neighbors
- Net effect: `5×center − (top + bottom + left + right)` → boosts edges and contrast
- This is equivalent to: `4×center − Laplacian` → sharpened output

---

## Filters Used in This Project

### 1. Gaussian Blur Filter

```python
def apply_blur(image, radius=5):
    return image.filter(ImageFilter.GaussianBlur(radius))
```

| Property      | Detail |
|---------------|--------|
| Method        | PIL `ImageFilter.GaussianBlur` |
| Parameter     | `radius=5` (controls spread/sigma) |
| Effect        | Softens image, removes noise |
| Use case      | Pre-processing, noise reduction, depth-of-field effect |

A **radius of 5** means the Gaussian window extends 5 pixels in each direction, producing a noticeably soft result.

---

### 2. PIL Sharpen Filter

```python
def apply_sharpen(image):
    return image.filter(ImageFilter.SHARPEN)
```

| Property  | Detail |
|-----------|--------|
| Method    | PIL `ImageFilter.SHARPEN` |
| Effect    | Mild edge enhancement |
| Kernel    | Built-in, approximately a 3×3 unsharp mask |
| Use case  | Quick sharpening with minimal configuration |

PIL's built-in `SHARPEN` filter applies a pre-defined unsharp masking kernel internally. It is less aggressive than custom kernel sharpening and suitable for a light enhancement pass.

---

### 3. Volvo8 Sharpen Filter

```python
def volvo8_sharpen(image):
    image = np.array(image)
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return Image.fromarray(sharpened)
```

| Property  | Detail |
|-----------|--------|
| Method    | `cv2.filter2D` with custom 3×3 kernel |
| Kernel    | Laplacian-based sharpening (`center=5`, cross neighbors=`-1`) |
| Effect    | Crisp edge enhancement using OpenCV's optimized convolution |
| Use case  | Strong edge sharpening for image clarity improvement |

This filter uses **OpenCV's `filter2D`** function, which is highly optimized for performance even on large images. The image is first converted to a NumPy array for kernel application, then converted back to a PIL Image.

The `ddepth=-1` argument in `cv2.filter2D` means the output has the same bit depth as the input.

---

### 4. General Sharpen Filter

```python
def general_sharpen(image):
    image = np.array(image)
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return Image.fromarray(sharpened)
```

| Property  | Detail |
|-----------|--------|
| Method    | `cv2.filter2D` with the same 3×3 kernel |
| Kernel    | Identical Laplacian-based kernel |
| Effect    | Same result as Volvo8 (useful as a baseline comparison) |
| Use case  | Demonstrates that the kernel, not the function name, defines the outcome |

This function intentionally uses the same kernel as Volvo8 to establish that sharpening behavior is determined entirely by the kernel weights. It can serve as a baseline for experimenting with alternative kernels.

---

## Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| `opencv-python` (`cv2`) | ≥ 4.x | Custom kernel convolution via `filter2D` |
| `Pillow` (`PIL`) | ≥ 9.x | Image loading, Gaussian blur, PIL sharpen |
| `NumPy` | ≥ 1.21 | Array conversion for kernel operations |
| `Matplotlib` | ≥ 3.x | Side-by-side image display |
| `google.colab` | Built-in | File upload and `cv2_imshow` support in Colab |

---

## Project Structure

```
image-processing-filters/
│
├── image_filters.py       # Main script with all filter functions
├── README.md              # Project documentation (this file)
└── sample_output.png      # Example output grid (optional)
```

---

## How to Run

### Step 1 — Open in Google Colab

Upload or copy the script into a new Google Colab notebook.

### Step 2 — Install dependencies (if needed)

Most libraries are pre-installed in Colab. If any are missing:

```bash
!pip install opencv-python pillow numpy matplotlib
```

### Step 3 — Run the script

Execute all cells in order. When the `files.upload()` cell runs, a file picker dialog will appear.

### Step 4 — Upload an image

Choose any `.jpg`, `.jpeg`, or `.png` image from your local machine.

### Step 5 — View results

A 2×3 grid will display:
1. Original Image
2. Blurred Image
3. PIL Sharpened Image
4. Volvo8 Sharpened Image
5. General Sharpened Image

---

## Output Description

The `display_images()` function renders a **2-row × 3-column subplot grid** using Matplotlib:

```
┌──────────────┬──────────────┬──────────────┐
│   Original   │   Blurred    │  PIL Sharpen │
├──────────────┼──────────────┼──────────────┤
│Volvo8 Sharpen│General Sharpen│    (empty)  │
└──────────────┴──────────────┴──────────────┘
```

Each subplot is displayed without axis ticks (`axis('off')`) for a clean visual comparison.

---

## Applications

| Domain | Application |
|--------|-------------|
| Medical Imaging | Sharpening X-rays or MRI scans to enhance diagnostic clarity |
| Photography | Noise reduction (blur) and detail recovery (sharpen) in post-processing |
| Surveillance | Enhancing blurry CCTV footage for object/face recognition |
| Satellite Imagery | Improving resolution of aerial photos |
| Document Scanning | Sharpening scanned text for better OCR accuracy |
| Machine Learning | Pre-processing images before feeding into CNN models |

---

## References

- Gonzalez, R. C., & Woods, R. E. — *Digital Image Processing*, 4th Edition, Pearson (2018)
- OpenCV Documentation — [cv2.filter2D](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04)
- Pillow Documentation — [ImageFilter Module](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html)
- Shapiro, L. G., & Stockman, G. C. — *Computer Vision*, Prentice Hall (2001)

---

> **Tip:** To experiment further, try replacing the kernel values in `volvo8_sharpen` or `general_sharpen` with other kernels such as edge detection (Sobel, Prewitt), emboss, or motion blur kernels to observe different effects.
