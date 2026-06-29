# LOG-02: Lựa chọn và Khóa Cấu trúc Mô hình Học máy Phân loại (ML Model Selection & Locking Justification)

- **Ngày ghi nhận:** 29/06/2026
- **Người thực hiện:** Nhóm nghiên cứu & AI Assistant
- **Trạng thái:** Đã phê duyệt (Approved & Locked)
- **Danh mục:** Kiến trúc Mô hình & Phương pháp luận (Model Architecture & Methodology)
- **Tham chiếu:** Kế thừa định hướng từ `LOG-01` (Research Scoping Decision)

---

## 1. BỐI CẢNH & QUYẾT ĐỊNH CỐ ĐỊNH MÔ HÌNH (CONTEXT & DECISION)

Để thực hiện chiến lược thu hẹp quy mô nghiên cứu sâu (Depth over Breadth) đã thống nhất tại `LOG-01`, nhóm nghiên cứu quyết định không khảo sát dàn trải các thuật toán phân loại học máy khác nhau và không thực hiện Grid Search mở rộng siêu tham số mô hình trong Phase 3.

Thay vào đó, nghiên cứu **khóa cố định (Lock) toàn bộ pipeline phân loại học máy** ở tiêu chuẩn vàng:

$$\text{Pipeline cố định: } \mathbf{StandardScaler()} \longrightarrow \mathbf{SVC}\big(\text{kernel='rbf'}, C=10.0, \gamma=\text{'scale'}, \text{class\_weight='balanced'}\big)$$

Việc khóa cố định này tạo ra một hệ quy chiếu đo lường tuyệt đối ổn định, giúp toàn bộ các sai lệch hay cải tiến về điểm số F1/Accuracy trong thực nghiệm phản ánh trung thực 100% chất lượng của các kỹ thuật Tiền xử lý ảnh (Data Preprocessing) và Trích xuất đặc trưng (Feature Extraction).

---

## 2. NỘI DUNG 1: TẠI SAO CHỌN VÀ KHÓA CỐ ĐỊNH MÔ HÌNH SVM? (WHY SVM IS CHOSEN & LOCKED)

Việc lựa chọn **Support Vector Machine (SVM)** làm mô hình cố định duy nhất được xây dựng trên 4 trụ cột toán học và thực nghiệm vượt trội cho bài toán biển báo giao thông Việt Nam (bộ dữ liệu NVTS):

### 2.1. Tương thích hoàn hảo với Không gian số chiều cao (High-Dimensional Space Synergy)
* **Bản chất dữ liệu:** Đặc trưng HOG sinh ra các vector có số chiều khổng lồ (từ $1,764\text{ chiều}$ đến $5,804\text{ chiều}$). Trong khi đó, số lượng mẫu huấn luyện của nhiều lớp biển báo thiểu số lại rất hạn chế (vài chục ảnh) $\rightarrow$ Bài toán **High-dimensional, Low-sample-size (HDLSS)** kinh điển.
* **Cơ chế miễn nhiễm Lời nguyền số chiều:** Dựa trên lý thuyết Tối ưu hóa lề tối đa (*Vapnik, 1995*), đường biên phân chia của SVM **chỉ phụ thuộc vào tập hợp các điểm hỗ trợ sát ranh giới nhất (Support Vectors)**. Số lượng Support Vectors là nhỏ và độc lập với số chiều, giúp SVM không bị suy giảm hiệu năng khi đối mặt với vector HOG nghìn chiều.

### 2.2. Sức mạnh Ánh xạ phi tuyến (Kernel Trick - RBF Kernel)
* Ranh giới phân chia biển báo ngoài trời (do xô lệch góc chụp 3D, phai màu trời mưa/nắng, bóng râm cây che) có cấu trúc phi tuyến tính cực kỳ phức tạp.
* Thủ thuật Kernel RBF chiếu ngầm vector đặc trưng lên không gian Hilbert chiều vô hạn, cho phép tạo ra các đường biên phân chia uốn lượn ôm sát từng cụm phân bố dữ liệu mà không tốn chi phí tính toán tọa độ mới.

### 2.3. Giảm thiểu Rủi ro Cấu trúc (Structural Risk Minimization - SRM)
* SVM tối ưu hóa Hinge Loss dựa trên nguyên lý **SRM**, cân bằng giữa giảm thiểu lỗi trên tập train và tối đa hóa vùng đệm an toàn (Margin). Khi kết hợp với tham số `class_weight='balanced'`, SVM tự động phạt nặng các lỗi phân loại trên lớp thiểu số, bảo vệ ranh giới công bằng cho các biển báo hiếm gặp tại Việt Nam.

### 2.4. Tính xác định & Độc lập khởi tạo (Convex Optimization)
* Hàm mục tiêu của SVM là bài toán **Quy hoạch toàn phương lồi (Convex Quadratic Programming)**. Quá trình huấn luyện **luôn hội tụ về nghiệm tối ưu toàn cục duy nhất (Global Optimum)**, không bị ảnh hưởng bởi khởi tạo ngẫu nhiên hay kẹt ở cực tiểu cục bộ. Điều này mang lại độ tin cậy khoa học tuyệt đối cho luận văn: mọi kết quả so sánh đều do chất lượng của đặc trưng mang lại.

### 2.5. Lý giải Thiết lập Bộ Siêu tham số Khóa (Locked Hyperparameters)
1. **`StandardScaler()`**: Ép mọi chiều đặc trưng về phân phối chuẩn ($\mu=0, \sigma=1$) để cân bằng khoảng cách Euclid giữa các đặc trưng khác dải số (Raw Pixels vs HOG vs Histogram).
2. **`kernel = 'rbf'`**: Tiêu chuẩn vàng phi tuyến cho thị giác máy tính cổ điển.
3. **`C = 10.0`**: Biển báo giao thông có ranh giới khắt khe (ví dụ: biển `40` vs `50`). Phạt lỗi $C=10.0$ tạo đường biên chặt chẽ, giảm margin errors để phân loại chính xác các chi tiết nhỏ.
4. **`gamma = 'scale'`**: Tự động tính toán theo công thức $\gamma = \frac{1}{n\_features \times \text{Var}(X)}$, giúp co giãn độ mở kernel tỷ lệ nghịch với số chiều nghìn chiều của HOG.
5. **`class_weight = 'balanced'`**: Gán trọng số phạt lỗi nghịch đảo với tần suất xuất hiện của lớp, giải quyết trực tiếp sự mất cân bằng dữ liệu đuôi dài thực tế.

---

## 3. NỘI DUNG 2: TẠI SAO CÁC MÔ HÌNH ML KHÁC KHÔNG ĐƯỢC CHỌN TRONG BỐI CẢNH NÀY? (WHY ALTERNATIVE CLASSIFICATION MODELS WERE REJECTED)

Mặc dù các mô hình dưới đây rất mạnh trong các bài toán phân loại bảng (tabular classification) nói chung, nhưng khi đặt vào bối cảnh nghiên cứu cụ thể này (Đặc trưng HOG nghìn chiều + Dữ liệu ảnh biển báo mất cân bằng), chúng để lộ những nhược điểm chí mạng:

### 3.1. Nhóm Cây quyết định & Boosting (Random Forest, XGBoost, LightGBM)
* **Cơ chế phân lớp:** Phân chia không gian bằng các đường cắt vuông góc với trục tọa độ (*axis-aligned orthogonal splits*).
* **Lý do loại trừ:** Đặc trưng HOG sinh ra các vector có tính chất gradient liên tục trong không gian $1,764 - 5,804\text{ chiều}$. Khi cắt vuông góc trên không gian này, ranh giới quyết định của cây bị phân mảnh bậc thang trầm trọng. Để phân loại chính xác, cây phải phát triển độ sâu khổng lồ, dẫn đến **Overfitting nặng nề trên tập Train** và **tốc độ suy luận (inference) cực chậm**, không đáp ứng được yêu cầu nhận diện biển báo thời gian thực.

### 3.2. Nhóm Mô hình Khoảng cách & Thống kê cơ bản (K-Nearest Neighbors - KNN, Naive Bayes)
* **Lý do loại trừ KNN:** Trong không gian nghìn chiều, KNN mắc phải hiện tượng cô đặc khoảng cách (*Distance Concentration*). Khoảng cách Euclid giữa điểm gần nhất và điểm xa nhất tiến tới bằng nhau, làm KNN mất hoàn toàn định hướng phân lớp.
* **Lý do loại trừ Naive Bayes:** Dựa trên giả định các đặc trưng độc lập có điều kiện (*Conditional Independence*). Tuy nhiên, các ô cell lân cận trong HOG hoặc các giỏ màu trong Histogram có sự tương quan hình học cực kỳ chặt chẽ. Vi phạm giả định độc lập khiến Naive Bayes tính toán sai lệch hoàn toàn xác suất hậu nghiệm.

### 3.3. Mô hình Tuyến tính Thống kê (Logistic Regression / Softmax Regression)
* **Lý do loại trừ:** 
  1. *Ranh giới tuyến tính:* Chỉ có thể vẽ các siêu phẳng phẳng ($w^T x + b = 0$), không thể ôm sát các cụm phân bố biển báo biến dạng phi tuyến ngoài trời $\rightarrow$ **Underfitting**.
  2. *Triết lý Log-loss vs. Hinge Loss:* Hàm Log-loss bị chi phối bởi toàn bộ điểm dữ liệu. Khi dữ liệu mất cân bằng đuôi dài, lực kéo khổng lồ của lớp đa số sẽ kéo lệch đường biên lấn sang lớp thiểu số.
  3. *Phân kỳ trọng số:* Trong không gian nghìn chiều có các cụm phân tách hoàn hảo (*Quasi-complete separation*), trọng số Logistic Regression bị phân kỳ tiến tới vô cùng ($\|w\| \rightarrow \infty$), gây Overfitting trầm trọng.

### 3.4. Mạng Nơ-ron Nhân tạo Truyền thống (Multilayer Perceptron - MLP / ANN)
* **Lý do loại trừ:** MLP tối ưu hóa hàm mất mát phi lồi (*Non-convex optimization*) thông qua Backpropagation, kết quả phụ thuộc lớn vào khởi tạo ngẫu nhiên và dễ bị kẹt ở cực tiểu cục bộ (*Local Minima*). Khi huấn luyện trên tập dữ liệu nhỏ nhưng số chiều lớn (HDLSS), MLP cực kỳ dễ overfit và khó kiểm soát. Ngược lại, SVM luôn bảo đảm nghiệm tối ưu toàn cục duy nhất, phù hợp để làm hệ quy chiếu chuẩn xác cho luận văn.
