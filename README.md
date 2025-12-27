# 📸 Image & Video Processing Laboratory

This project is a comprehensive desktop application developed for the **Computer Engineering Computer Vision** course. It incorporates techniques for both dynamic (video) and static (still image) processing and was engineered using **PyQt5** and **OpenCV**.

## 🚀 Features

The application is composed of two primary modules:

### 🎥 Video Studio
* **Motion Analysis:** Optical Flow implementation using the Lucas-Kanade Method.
* **Object Tracking:** Real-time tracking utilizing KCF, CSRT, and MeanShift algorithms.
* **Video Editing:**
    * Video Reversal
    * Multi-ROI (Region of Interest) Cropping
    * Temporal Trimming (Time-Interval Cutting)
    * Regional Blurring
* **Analysis:** Histogram-based scene change detection and graphical plotting.
* **Filters:** RGB channel control, Gamma correction, and Grayscale conversion.

### 📷 Photo Studio
* **Geometric Transformations:** Rotation, Scaling (Resizing), and Mirroring (Flipping).
* **Fundamental Adjustments:** Control of RGB color channels and Gamma (Luminance) correction.
* **Advanced Filters:**
    * **Edge Detection:** Canny and Sobel operators.
    * **Noise Reduction:** Morphological operations (Erosion and Dilation).
    * Blurring and Sharpening.
* **Analysis:** RGB color histogram analysis (integrated with Matplotlib).

## 🛠️ Installation and Usage

1.  **Clone or download** the repository.
2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch the application:**
    ```bash
    python main.py
    ```

## 📚 Technologies Used
* **Language:** Python 3.x
* **GUI Framework:** PyQt5
* **Image Processing:** OpenCV (cv2)
* **Numerical Computation:** NumPy
* **Data Visualization:** Matplotlib

---
*This project has been developed for educational purposes.*