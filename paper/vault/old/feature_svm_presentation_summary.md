# BẢN TIẾNG VIỆT (VIETNAMESE VERSION)

Tổng hợp đầy đủ và chi tiết các thông tin được trích xuất trực tiếp từ notebook `exphase_1.ipynb`, được bố trí theo cấu trúc chuẩn để bạn dễ dàng đưa vào **slide thuyết trình**:

---

## PHẦN 1: CẤU HÌNH DỮ LIỆU & TIỀN XỬ LÝ (DATA CONFIG)

1. **Bộ dữ liệu gốc**: `cropped_dataset` (chia sẵn thành 3 tập `train`, `val`, `test`).
2. **Kích thước ảnh mục tiêu (`TARGET_SIZE`)**: `64x64 pixel`. 
   * *Tất cả ảnh trong bộ dữ liệu đều được bắt buộc đưa về cùng kích thước 64x64 trước khi trích xuất đặc trưng.*
3. **Chiến lược nội suy khi Resize (`Interpolation Method`)**:
   * **`cv2.INTER_AREA`**: Sử dụng khi ảnh gốc lớn hơn 64x64 (thu nhỏ ảnh giúp tránh mất mát thông tin cục bộ).
   * **`cv2.INTER_CUBIC`**: Sử dụng khi ảnh gốc nhỏ hơn 64x64 (phóng to ảnh rõ nét hơn).
4. **Định dạng ảnh hỗ trợ**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.
5. **Thống kê phân bổ dữ liệu**:
   * **Train split**: 6,605 ảnh $\rightarrow$ Tensor shape: `(6605, 64, 64, 3)`
   * **Validation split**: 824 ảnh $\rightarrow$ Tensor shape: `(824, 64, 64, 3)` *(dùng để chọn ra bộ đặc trưng tốt nhất)*
   * **Test split**: 851 ảnh $\rightarrow$ Tensor shape: `(851, 64, 64, 3)` *(chỉ đánh giá 1 lần duy nhất ở bước cuối)*

---

## PHẦN 2: CẤU HÌNH PHÂN LOẠI & MÔ HÌNH SVM (SVM MODEL CONFIG)

Trong toàn bộ các thí nghiệm, pipeline phân loại được giữ cố định hoàn toàn nhằm đảm bảo tính công bằng khi so sánh các bộ đặc trưng:

$$\text{Pipeline: } \mathbf{StandardScaler()} \longrightarrow \mathbf{SVC()}$$

* **Bước chuẩn hóa (`StandardScaler`)**: Chuẩn hóa từng chiều đặc trưng về phân phối chuẩn có $\mu = 0, \sigma = 1$.
* **Các tham số của Support Vector Machine (`SVM_PARAMS`)**:
  * **`kernel`**: `'rbf'` *(Radial Basis Function)*
  * **`C`** *(Regularization parameter)*: `10.0` *(chấp nhận đường biên phức tạp để giảm lỗi phân loại)*
  * **`gamma`**: `'scale'` *(được tính tự động theo công thức $\frac{1}{n\_features \times \text{Var}(X)}$)*
  * **`class_weight`**: `'balanced'` *(tự động điều chỉnh trọng số tỷ lệ nghịch với tần suất lớp, khắc phục mất cân bằng dữ liệu)*
  * **`cache_size`**: `1024 MB` *(tăng tốc độ tính toán ma trận kernel)*

---

## PHẦN 3: BỘ SIÊU THAM SỐ TRÍCH XUẤT ĐẶC TRƯNG CHUNG

1. **Siêu tham số HOG (`HOG_PARAMS`)**:
   * **`orientations`**: `9` *(9 hướng gradient từ $0^\circ$ đến $180^\circ$)*
   * **`pixels_per_cell`**: `(8, 8)` *(mỗi ô cell có kích thước 8x8 pixel)*
   * **`cells_per_block`**: `(2, 2)` *(mỗi block chuẩn hóa gồm 2x2 cells, tức 16x16 pixel)*
   * **`block_norm`**: `'L2-Hys'` *(L2-Hysteresis - chuẩn hóa theo block giúp bất biến với sự thay đổi ánh sáng)*
   * **`transform_sqrt`**: `True` *(áp dụng căn bậc hai lên ảnh gốc trước khi tính gradient để giảm nhiễu chói/bóng râm)*
   * **`feature_vector`**: `True` *(trả về vector 1D)*

2. **Siêu tham số Color Histogram (`COLOR_HIST_BINS`)**:
   * **`bins`**: `(8, 8, 8)` trên 3 kênh màu HSV *(Hue: 8 bins, Saturation: 8 bins, Value: 8 bins)* $\rightarrow$ Tổng số chiều = $8 \times 8 \times 8 = \mathbf{512\text{ chiều}}$.
   * **Chuẩn hóa**: Chia histogram cho tổng các phần tử để biến thành mật độ xác suất `(hist /= hist.sum() + 1e-8)`.

---

## PHẦN 4: CHI TIẾT PHƯƠNG PHÁP & THAM SỐ CỦA 4 NHÓM ĐƯỢC YÊU CẦU

### 1. Nhóm "HOG (Gray)"
* **Phương pháp trích xuất (`Feature Extraction Method`)**:
  1. Chuyển ảnh BGR sang hệ màu xám Grayscale `(cv2.COLOR_BGR2GRAY)`.
  2. Chuẩn hóa giá trị pixel từ `[0, 255]` về khoảng kiểu số thực `[0.0, 1.0]` bằng cách chia cho `255.0`.
  3. Trích xuất đặc trưng HOG trên kênh xám này.
* **Tính toán số chiều đặc trưng (`n_features`)**:
  * Với ảnh 64x64 và `pixels_per_cell=(8,8)` $\rightarrow$ lưới ô gồm $8 \times 8$ cells.
  * Bước trượt block 2x2 cells $\rightarrow$ có $(8 - 2 + 1) \times (8 - 2 + 1) = 7 \times 7 = \mathbf{49\text{ blocks}}$.
  * Mỗi block chứa $2 \times 2 \times 9 = 36\text{ giá trị}$.
  * **Tổng số chiều vector**: $49 \times 36 = \mathbf{1,764\text{ chiều}}$.
* **Kết quả trên tập Validation**: `Accuracy` = **95.27%** | `Macro F1` = **92.86%** *(Thời gian train: 29.6s)*

### 2. Nhóm "HOG (Gray) + Color Histogram (HSV)"  $\star$ *(MÔ HÌNH TỐT NHẤT)*
* **Phương pháp trích xuất (`Feature Extraction Method`)**:
  1. Tính vector đặc trưng **HOG (Gray)** (1,764 chiều) từ ảnh Grayscale.
  2. Tính vector **Color Histogram 3D** (512 chiều) từ ảnh chuyển sang hệ màu HSV `(cv2.COLOR_BGR2HSV)`.
  3. Nối tiếp 2 vector theo chiều ngang (`np.hstack`).
* **Số chiều đặc trưng (`n_features`)**: $1,764 + 512 = \mathbf{2,276\text{ chiều}}$.
* **Kết quả trên tập Validation**: `Accuracy` = **95.51%** | `Macro F1` = **93.27%** *(Thời gian train: 40.2s)*
  * *Lưu ý thuyết trình:* Đây chính là sự kết hợp **đạt hiệu năng cao nhất trên tập Validation**, tận dụng được cả *đặc trưng hình học/đường nét (HOG)* lẫn *đặc trưng màu sắc bất biến ánh sáng (HSV)*. Khi đánh giá lần cuối trên tập **Test**, mô hình đạt **Test Accuracy = 93.07%** và **Test Macro F1 = 91.21%**.

### 3. Nhóm "HOG (YUV)"
* **Phương pháp trích xuất (`Feature Extraction Method`)**:
  1. Chuyển ảnh BGR sang không gian màu YUV `(cv2.COLOR_BGR2YUV)` *(Y: độ sáng Luma; U, V: thông tin màu Chrominance)*.
  2. Tách độc lập 3 kênh Y, U, V và chuẩn hóa từng kênh về `[0.0, 1.0]`.
  3. Trích xuất HOG riêng biệt trên từng kênh một, sau đó nối 3 vector kết quả lại với nhau (`np.concatenate`).
* **Số chiều đặc trưng (`n_features`)**: $1,764 \times 3\text{ kênh} = \mathbf{5,292\text{ chiều}}$.
* **Kết quả trên tập Validation**: `Accuracy` = **95.51%** | `Macro F1` = **92.74%** *(Thời gian train: 135.8s)*

### 4. Nhóm "HOG (YUV) + Color Histogram (HSV)"
* **Phương pháp trích xuất (`Feature Extraction Method`)**:
  1. Trích xuất cụm đặc trưng **HOG (YUV)** trên 3 kênh Y, U, V (5,292 chiều).
  2. Trích xuất **Color Histogram (HSV)** (512 chiều).
  3. Nối ghép nối tiếp hai mảng đặc trưng lại (`np.hstack`).
* **Số chiều đặc trưng (`n_features`)**: $5,292 + 512 = \mathbf{5,804\text{ chiều}}$ *(bộ đặc trưng nặng nhất)*.
* **Kết quả trên tập Validation**: `Accuracy` = **95.51%** | `Macro F1` = **92.09%** *(Thời gian train: 142.9s)*

> [!TIP]
> **Gợi ý điểm nhấn phản biện cho bài thuyết trình:**
> * Mặc dù nhóm **HOG (YUV)** và **HOG (YUV) + Color Hist** có số chiều cực lớn (>5000 chiều) và tốn thời gian train gấp ~3.5 lần, hiệu năng phân loại (`Macro F1`) lại *thấp hơn* so với việc chỉ dùng **HOG (Gray) + Color Hist (HSV)** (2,276 chiều).
> * Lý do: Kênh Y trong YUV đã chứa thông tin độ sáng tương tự Grayscale, trong khi gradient trên các kênh màu U, V thường khá nhiễu và dễ gây hiện tượng *curse of dimensionality* (lời nguyền số chiều) cho mô hình SVM.

---
---

# ENGLISH VERSION

Comprehensive and structured summary extracted directly from `exphase_1.ipynb`, optimized for your **presentation slides**:

---

## PART 1: DATA CONFIGURATION & PREPROCESSING

1. **Source Dataset**: `cropped_dataset` (pre-divided into `train`, `val`, and `test` splits).
2. **Target Image Size (`TARGET_SIZE`)**: `64x64 pixels`.
   * *Every input image is strictly resized to 64x64 pixels prior to feature extraction.*
3. **Interpolation Strategy during Resizing**:
   * **`cv2.INTER_AREA`**: Applied when downsampling (image dimensions > 64x64) to prevent moiré patterns and preserve local structural details.
   * **`cv2.INTER_CUBIC`**: Applied when upsampling (image dimensions < 64x64) for sharper edge rendering.
4. **Supported File Extensions**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.
5. **Dataset Split Distribution**:
   * **Train split**: 6,605 images $\rightarrow$ Tensor shape: `(6605, 64, 64, 3)`
   * **Validation split**: 824 images $\rightarrow$ Tensor shape: `(824, 64, 64, 3)` *(used for model selection)*
   * **Test split**: 851 images $\rightarrow$ Tensor shape: `(851, 64, 64, 3)` *(used strictly once for final benchmark)*

---

## PART 2: CLASSIFIER & SVM MODEL CONFIGURATION

Across all experiments, the classification pipeline is kept identical to ensure a fair comparison of feature spaces:

$$\text{Pipeline: } \mathbf{StandardScaler()} \longrightarrow \mathbf{SVC()}$$

* **Feature Scaling (`StandardScaler`)**: Standardizes features by removing the mean and scaling to unit variance ($\mu = 0, \sigma = 1$).
* **Support Vector Machine Hyperparameters (`SVM_PARAMS`)**:
  * **`kernel`**: `'rbf'` *(Radial Basis Function)*
  * **`C`** *(Regularization parameter)*: `10.0` *(lower bias / tighter decision boundary tolerance)*
  * **`gamma`**: `'scale'` *(automatically computed as $\frac{1}{n\_features \times \text{Var}(X)}$)*
  * **`class_weight`**: `'balanced'` *(inversely proportional to class frequencies to mitigate dataset imbalance)*
  * **`cache_size`**: `1024 MB` *(increases kernel matrix computation efficiency)*

---

## PART 3: GENERAL FEATURE EXTRACTION HYPERPARAMETERS

1. **HOG Hyperparameters (`HOG_PARAMS`)**:
   * **`orientations`**: `9` *(gradient direction bins spaced evenly between $0^\circ$ and $180^\circ$)*
   * **`pixels_per_cell`**: `(8, 8)` *(spatial cell resolution of 8x8 pixels)*
   * **`cells_per_block`**: `(2, 2)` *(local contrast normalization blocks of 2x2 cells, i.e., 16x16 pixels)*
   * **`block_norm`**: `'L2-Hys'` *(L2-Hysteresis block normalization for illumination invariance)*
   * **`transform_sqrt`**: `True` *(power law / gamma compression to dampen heavy shadow and highlight variations)*
   * **`feature_vector`**: `True` *(returns flattened 1D array)*

2. **Color Histogram Hyperparameters (`COLOR_HIST_BINS`)**:
   * **`bins`**: `(8, 8, 8)` across 3 HSV color channels *(Hue: 8 bins, Saturation: 8 bins, Value: 8 bins)* $\rightarrow$ Total feature dimensionality = $8 \times 8 \times 8 = \mathbf{512\text{ dimensions}}$.
   * **Normalization**: L1 normalized by dividing by the bin sum `(hist /= hist.sum() + 1e-8)`.

---

## PART 4: DETAILED METHODS & VARIANTS FOR THE 4 REQUESTED GROUPS

### 1. Group "HOG (Gray)"
* **Feature Extraction Method**:
  1. Convert BGR images to Grayscale `(cv2.COLOR_BGR2GRAY)`.
  2. Normalize pixel intensities from `[0, 255]` to floating-point range `[0.0, 1.0]` by dividing by `255.0`.
  3. Extract HOG descriptors on the single grayscale channel.
* **Feature Dimensionality (`n_features`)**:
  * For 64x64 images and `pixels_per_cell=(8,8)` $\rightarrow$ grid consists of $8 \times 8$ cells.
  * Sliding block window of 2x2 cells $\rightarrow$ $(8 - 2 + 1) \times (8 - 2 + 1) = 7 \times 7 = \mathbf{49\text{ blocks}}$.
  * Each block produces $2 \times 2 \times 9 = 36\text{ values}$.
  * **Total Vector Length**: $49 \times 36 = \mathbf{1,764\text{ dimensions}}$.
* **Validation Performance**: `Accuracy` = **95.27%** | `Macro F1` = **92.86%** *(Training time: 29.6s)*

### 2. Group "HOG (Gray) + Color Histogram (HSV)"  $\star$ *(BEST PERFORMING MODEL)*
* **Feature Extraction Method**:
  1. Compute **HOG (Gray)** descriptor (1,764 dimensions).
  2. Compute **3D Color Histogram** (512 dimensions) from images transformed to HSV space `(cv2.COLOR_BGR2HSV)`.
  3. Horizontally concatenate both feature vectors (`np.hstack`).
* **Feature Dimensionality (`n_features`)**: $1,764 + 512 = \mathbf{2,276\text{ dimensions}}$.
* **Validation Performance**: `Accuracy` = **95.51%** | `Macro F1` = **93.27%** *(Training time: 40.2s)*
  * *Key Presentation Highlight:* This specific combination achieves the **highest overall classification performance on the Validation split**, successfully pairing *shape/edge structural descriptors (HOG)* with *illumination-invariant color distributions (HSV)*. On the final **Test benchmark**, this model scores **Test Accuracy = 93.07%** and **Test Macro F1 = 91.21%**.

### 3. Group "HOG (YUV)"
* **Feature Extraction Method**:
  1. Convert BGR images to YUV color space `(cv2.COLOR_BGR2YUV)` *(Y: Luma/brightness; U, V: Chrominance/color)*.
  2. Separate the 3 channels independently and scale each to `[0.0, 1.0]`.
  3. Extract HOG features separately on each channel, then concatenate the 3 resulting vectors (`np.concatenate`).
* **Feature Dimensionality (`n_features`)**: $1,764 \times 3\text{ channels} = \mathbf{5,292\text{ dimensions}}$.
* **Validation Performance**: `Accuracy` = **95.51%** | `Macro F1` = **92.74%** *(Training time: 135.8s)*

### 4. Group "HOG (YUV) + Color Histogram (HSV)"
* **Feature Extraction Method**:
  1. Compute **HOG (YUV)** multi-channel descriptors (5,292 dimensions).
  2. Compute **Color Histogram (HSV)** (512 dimensions).
  3. Concatenate both representations (`np.hstack`).
* **Feature Dimensionality (`n_features`)**: $5,292 + 512 = \mathbf{5,804\text{ dimensions}}$ *(highest dimensionality)*.
* **Validation Performance**: `Accuracy` = **95.51%** | `Macro F1` = **92.09%** *(Training time: 142.9s)*

> [!TIP]
> **Key Discussion Points for Presentation Q&A:**
> * Despite **HOG (YUV)** and **HOG (YUV) + Color Hist** having massive feature spaces (>5,000 dimensions) and requiring ~3.5x longer training times, their classification effectiveness (`Macro F1`) is *inferior* to the much more compact **HOG (Gray) + Color Hist (HSV)** (2,276 dimensions).
> * Rationale: The Y channel in YUV already encapsulates brightness information equivalent to Grayscale, whereas gradients calculated across chrominance channels (U, V) tend to introduce noisy gradient vectors, leading to the *curse of dimensionality* for the RBF SVM kernel.
