# BẢN TIẾNG VIỆT (VIETNAMESE VERSION)

Dưới đây là bản **cụ thể hóa các hướng cải thiện** từ `finetuning.md`, biến các ý tưởng chung thành **đề xuất kỹ thuật chi tiết có công thức, dẫn chứng số liệu thực nghiệm và giải pháp pipeline** để bạn gây ấn tượng mạnh trước hội đồng phản biện:

---

## 1. KHẮC PHỤC BIẾN DẠNG HÌNH HỌC VẬT THỂ (ASPECT-RATIO PRESERVING PADDING)

* **Vấn đề thực nghiệm**: Trong `exphase_1.ipynb`, hàm `cv2.resize(img, (64, 64))` ép buộc mọi bounding box được crop (vốn có hình chữ nhật dài như xe tải, biển báo đứng) thành hình vuông. Điều này làm **bóp méo tỷ lệ khung hình (Aspect Ratio)**, dẫn đến các góc gradient của HOG bị sai lệch hoàn toàn so với thực tế.
* **Đề xuất thuyết trình cụ thể**: 
  * Áp dụng kỹ thuật **Letterbox Padding** trước khi đưa vào HOG: Giữ nguyên tỷ lệ ảnh gốc $W \times H$, thu phóng sao cho cạnh dài nhất bằng $64$, cạnh còn lại được đệm viền đen (`cv2.copyMakeBorder`) để đạt chuẩn $64 \times 64$.
  * **Minh chứng logic**: Giúp bộ đường nét HOG của xe tải hay biển báo không bị "lùn đi" hoặc "bè ra", giữ nguyên hướng vector gradient $0^\circ - 180^\circ$.

---

## 2. CHUẨN HÓA ĐỘC LẬP KHI GHÉP NỐI ĐẶC TRƯNG (INDEPENDENT MULTIMODAL FUSION)

* **Vấn đề thực nghiệm**: Nhóm `HOG + Color Histogram` nối tiếp mảng theo chiều ngang (`np.hstack([hog, hist])`). Vector HOG có giá trị mật độ gradient rất nhỏ (thường $\in [0, 0.2]$), trong khi Color Histogram HSV là tỷ lệ xác suất giỏ màu $\in [0, 1.0]$. Việc đưa thẳng mảng gộp vào `StandardScaler()` sẽ khiến kernel RBF của SVM **bị lấn át bởi phân phối có phương sai lớn hơn**.
* **Đề xuất thuyết trình cụ thể**:
  * Tách biệt bước tiền xử lý bằng **`ColumnTransformer` / `FeatureUnion`**:
    1. **Luồng HOG**: Áp dụng `L2-Normalization` riêng cho mảng HOG.
    2. **Luồng Histogram**: Áp dụng `PowerTransformer (Yeo-Johnson)` hoặc `MinMaxScaler` riêng cho Histogram để kéo dải xác suất về phân phối chuẩn.
  * **Minh chứng logic**: Đảm bảo khoảng cách Euclid $\|x_i - x_j\|^2$ trong hàm kernel $K(x, y) = \exp(-\gamma \|x - y\|^2)$ nhận đóng góp trọng số công bằng từ cả đường nét lẫn màu sắc.

---

## 3. TỐI ƯU HÓA KÊNH YUV & GIẢM NHIỄU MÀU (WEIGHTED YUV HOG STRATEGY)

* **Vấn đề thực nghiệm**: Nghịch lý ở Phase 1 cho thấy mô hình nặng nhất `HOG (YUV) + Hist` (5,804 chiều) lại có `Macro F1 = 92.09%`, **thấp hơn** `HOG (Gray) + Hist` (2,276 chiều - `Macro F1 = 93.27%`). Nguyên nhân là kênh Y (Luma) đã phản ánh đủ hình khối, còn việc tính gradient trên kênh sắc độ U và V tạo ra rất nhiều vector nhiễu.
* **Đề xuất thuyết trình cụ thể**:
  * Thay vì cào bằng 3 kênh Y, U, V (mỗi kênh sinh ra 1,764 features), ta áp dụng **Chiến lược bất đối xứng (Asymmetric Resolution)**:
    * **Kênh Y (Khối độ sáng)**: Giữ độ phân giải cao `pixels_per_cell=(8,8), orientations=9` (1,764 chiều).
    * **Kênh U, V (Màu sắc)**: Giảm độ phân giải lưới xuống `pixels_per_cell=(16,16)` hoặc chỉ lấy `orientations=4`, hoặc áp dụng hệ số suy giảm $\alpha = 0.3$ trước khi nối:
      $$\mathbf{v}_{final} = \Big[ \mathbf{v}_Y,\; 0.3\mathbf{v}_U,\; 0.3\mathbf{v}_V,\; \mathbf{v}_{HSV} \Big]$$
  * **Minh chứng logic**: Loại bỏ tới ~50% số chiều nhiễu, khắc phục hiện tượng *Curse of Dimensionality* (Lời nguyền số chiều) cho SVM.

---

## 4. GIẢM CHIỀU & CHỐNG OVERFITTING (DIMENSIONALITY REDUCTION VIA PCA/LDA)

* **Vấn đề thực nghiệm**: Tập Train có $6,605$ ảnh nhưng bộ đặc trưng HOG YUV + Hist lên tới $5,804$ chiều $\rightarrow$ Tỷ lệ mẫu trên số chiều $\approx 1.14$, rủi ro overfitting cực kỳ nghiêm trọng và thời gian train bị kéo dài tới $142.9$ giây.
* **Đề xuất thuyết trình cụ thể**:
  * Chèn thêm bước **Principal Component Analysis (`PCA`)** giữ lại $95\%$ phương sai (variance), hoặc **Linear Discriminant Analysis (`LDA`)** vào trước `SVC()`.
  * **Minh chứng logic**: Kỹ thuật này giúp nén mảng từ $5,804$ chiều xuống chỉ còn $\sim 300 - 500$ thành phần chính. Thời gian train mô hình dự kiến sẽ **giảm từ 142.9s xuống dưới 5 giây** mà vẫn tăng được điểm Test F1 do đã lọc sạch nhiễu nền.

---

## 5. XỬ LÝ TRIỆT ĐỂ MẤT CÂN BẰNG LỚP (IMBALANCE RESOLUTION VIA HYBRID SMOTE)

* **Vấn đề thực nghiệm**: Báo cáo phân loại trên tập Test (`test_confusions`) cho thấy mô hình đạt `Test Accuracy = 93.07%` nhưng `Test Macro F1` chỉ đạt `91.21%`. Cụ thể, các lớp thiểu số bị nhận diện rất kém: lớp `W.205a` (Recall = 25%), lớp `P.103a` (Recall = 50%), lớp `I.409` (Recall = 67%). Việc thiết lập `class_weight='balanced'` là chưa đủ.
* **Đề xuất thuyết trình cụ thể**:
  * Kết hợp **Tiền xử lý 2 cấp độ**:
    1. **Cấp độ ảnh (Photometric/Geometric Augmentation)**: Dùng thư viện `Albumentations` xoay nhẹ $\pm 15^\circ$, thêm nhiễu Gaussian và lật ngang để tạo thêm mẫu cho các class có dưới $50$ ảnh train.
    2. **Cấp độ đặc trưng (Feature space SMOTE)**: Sử dụng `SVMSMOTE` hoặc `BorderlineSMOTE` từ thư viện `imbalanced-learn` tạo mẫu nội suy nội bộ ngay trên không gian vector HOG trước khi fit SVM.

---

## 6. BƯỚC TIẾN SANG DEEP LEARNING (END-TO-END REPRESENTATION LEARNING)

* **Vấn đề thực nghiệm**: Hand-crafted features (HOG, HSV Hist) chạm trần hiệu năng do con người phải tự định nghĩa các quy tắc cố định (ví dụ: chia giỏ màu 8x8x8 hay ô cell 8x8 là hoàn toàn mang tính cảm tính).
* **Đề xuất thuyết trình cụ thể**:
  * So sánh mô hình **2 Bước truyền thống** (Feature Extraction $\rightarrow$ Classifier) với **Học biểu diễn End-to-End** bằng CNN nhẹ (`MobileNetV3-Small` hoặc `ResNet18`).
  * **Minh chứng logic**: Các bộ lọc (Kernel Tích chập) trong tầng Convolution đóng vai trò như những "bộ trích xuất HOG khả vi", tự động điều chỉnh trọng số thông qua Backpropagation để nhận diện chính xác đường nét đặc trưng của từng biển báo/phương tiện.

---
---

# ENGLISH VERSION

Below is the **concrete engineering expansion** of `finetuning.md`, transforming high-level suggestions into **actionable technical proposals equipped with mathematical formulations, empirical evidence, and pipeline architectures** for your presentation slides:

---

## 1. GEOMETRIC DISTORTION MITIGATION (ASPECT-RATIO PRESERVING PADDING)

* **Empirical Flaw**: In `exphase_1.ipynb`, standard `cv2.resize(img, (64, 64))` forces non-square cropped bounding boxes (e.g., elongated trucks, vertical signs) into a 1:1 square. This introduces severe **Aspect Ratio Distortion**, skewing HOG gradient orientations away from true physical contours.
* **Concrete Presentation Proposal**: 
  * Implement **Letterbox Padding** prior to HOG extraction: Maintain original image dimensions $W \times H$, scale the longest edge to $64$, and symmetrically pad the remaining edge with zero-intensity pixels (`cv2.copyMakeBorder`) to achieve an unskewed $64 \times 64$ canvas.
  * **Engineering Rationale**: Prevents structural gradient vectors ($0^\circ - 180^\circ$) of vehicles and road signs from being artificially flattened or stretched.

---

## 2. INDEPENDENT MULTIMODAL FEATURE NORMALIZATION

* **Empirical Flaw**: The `HOG + Color Histogram` group performs direct horizontal concatenation (`np.hstack([hog, hist])`). HOG vectors contain small gradient density values (typically $\in [0, 0.2]$), whereas HSV Color Histograms represent probability distributions $\in [0, 1.0]$. Feeding unscaled fused arrays into `StandardScaler()` causes the RBF kernel to be **dominated by the feature modality with larger raw variance**.
* **Concrete Presentation Proposal**:
  * Decouple preprocessing using **`ColumnTransformer` / `FeatureUnion`**:
    1. **HOG Pipeline**: Apply dedicated `L2-Normalization` to structural descriptors.
    2. **Histogram Pipeline**: Apply dedicated `PowerTransformer (Yeo-Johnson)` or `MinMaxScaler` to map color probabilities into standard Gaussian distributions.
  * **Engineering Rationale**: Ensures the Euclidean distance $\|x_i - x_j\|^2$ inside the RBF kernel formula $K(x, y) = \exp(-\gamma \|x - y\|^2)$ receives balanced weighting from both shape contours and chromaticity.

---

## 3. CHROMATIC NOISE SUPPRESSION (WEIGHTED YUV HOG STRATEGY)

* **Empirical Flaw**: Phase 1 benchmark reveals an academic paradox: the highest-dimensional feature set `HOG (YUV) + Hist` (5,804 dims) scores `Macro F1 = 92.09%`, which is **inferior** to `HOG (Gray) + Hist` (2,276 dims - `Macro F1 = 93.27%`). The Y channel already captures full structural luminance, making gradients computed across chrominance channels (U, V) redundant and noisy.
* **Concrete Presentation Proposal**:
  * Replace uniform multi-channel extraction with an **Asymmetric Resolution Strategy**:
    * **Y Channel (Luma Block)**: High spatial grid resolution `pixels_per_cell=(8,8), orientations=9` (1,764 dims).
    * **U, V Channels (Chroma Blocks)**: Coarse spatial grid `pixels_per_cell=(16,16)` or reduced `orientations=4`, or apply an attenuation scalar $\alpha = 0.3$ prior to fusion:
      $$\mathbf{v}_{final} = \Big[ \mathbf{v}_Y,\; 0.3\mathbf{v}_U,\; 0.3\mathbf{v}_V,\; \mathbf{v}_{HSV} \Big]$$
  * **Engineering Rationale**: Eliminates ~50% of uninformative feature dimensions, directly counteracting the *Curse of Dimensionality*.

---

## 4. SUBSPACE COMPRESSION & OVERFITTING PREVENTION (PCA/LDA)

* **Empirical Flaw**: The training corpus contains $6,605$ samples, yet the HOG YUV + Hist feature matrix spans $5,804$ dimensions $\rightarrow$ Sample-to-dimension ratio $\approx 1.14$. This extreme high-dimensional regime causes overfitting and inflates training latency to $142.9$ seconds.
* **Concrete Presentation Proposal**:
  * Integrate **Principal Component Analysis (`PCA`)** retaining $95\%$ cumulative variance, or **Linear Discriminant Analysis (`LDA`)** immediately preceding `SVC()`.
  * **Engineering Rationale**: Compresses the $5,804$-dimensional space into $\sim 300 - 500$ orthogonal principal components. Expected training latency will **drop from 142.9s to under 5 seconds** while simultaneously improving Test Macro F1 by filtering out background noise.

---

## 5. SEVERE CLASS IMBALANCE RESOLUTION (HYBRID AUGMENTATION & SMOTE)

* **Empirical Flaw**: Final Test evaluation (`test_confusions`) demonstrates `Test Accuracy = 93.07%` but a lagging `Test Macro F1 = 91.21%`. Minority classes suffer severe misclassification: class `W.205a` (Recall = 25%), class `P.103a` (Recall = 50%), class `I.409` (Recall = 67%). Relying solely on `class_weight='balanced'` is insufficient.
* **Concrete Presentation Proposal**:
  * Deploy a **Two-Tier Hybrid Pipeline**:
    1. **Image-Level Augmentation**: Utilize `Albumentations` (RandomRotate $\pm 15^\circ$, Gaussian Noise, Horizontal Flip) to synthesize training images for minority classes containing $<50$ samples.
    2. **Feature-Level SMOTE**: Apply `SVMSMOTE` or `BorderlineSMOTE` (`imbalanced-learn`) directly on extracted HOG feature spaces prior to fitting the decision boundary.

---

## 6. PARADIGM SHIFT TO DEEP LEARNING (END-TO-END REPRESENTATION LEARNING)

* **Empirical Flaw**: Hand-crafted feature pipelines (HOG + HSV) reach a performance plateau due to rigid, human-defined heuristic constraints (e.g., arbitrarily fixing 8x8x8 color bins or 8x8 pixel cells).
* **Concrete Presentation Proposal**:
  * Contrast traditional **Two-Stage Pipelines** (Feature Extraction $\rightarrow$ Classifier) against **End-to-End Deep Learning** utilizing lightweight CNNs (`MobileNetV3-Small` or `ResNet18`).
  * **Engineering Rationale**: Convolutional kernels act as "differentiable, learnable HOG extractors", automatically optimizing spatial filter weights via Backpropagation to maximize inter-class separation.
