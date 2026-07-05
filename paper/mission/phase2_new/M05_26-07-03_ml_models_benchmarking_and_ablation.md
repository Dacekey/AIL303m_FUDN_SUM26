# NHIỆM VỤ M05: Thực thi Khảo sát & Đối chứng Các Mô hình Học máy Phân loại (Classical ML Models Benchmarking & Ablation Study)

- **Mã nhiệm vụ:** `M05`
- **Ngày ban hành:** 03/07/2026
- **Người thực hiện phụ trách:** Thành viên 4 (ML & Pipeline Engineer) *(Nhiệm vụ độc lập 1 người thực hiện)*
- **Giai đoạn thực hiện:** Giai đoạn Mở rộng / Thực nghiệm Đối chứng (Phase 2)
- **Đầu ra bắt buộc (Deliverables):** 
  1. Bảng số liệu đối chứng toàn diện 6 họ thuật toán ML (`M05_ml_models_comparison_results.csv`).
  2. Biểu đồ trực quan hóa Trade-off (Độ chính xác vs. Tốc độ) & Phân tích hiệu năng trên 10 lớp thiểu số.
  3. Báo cáo phân tích bản chất toán học/hình học giải thích sâu sắc lý do mô hình này vượt trội hay thua kém mô hình kia.

---

## 1. MỤC ĐÍCH & TẦM QUAN TRỌNG CỦA NHIỆM VỤ

Tại **Phase 1 (Giai đoạn Thực nghiệm Tối ưu hóa Pipeline - M01 đến M04)**, nhóm nghiên cứu đã tuân thủ nghiêm ngặt quyết định chiến lược [LOG-02](file:///home/dacekey/AIL303_SUM26/paper/vault/log/02_26-06-29_ml_model_selection_and_locking.md): *Khóa cố định bộ phân loại ở tiêu chuẩn vàng `StandardScaler + SVC(kernel='rbf', C=10.0)`* để tập trung tối ưu hóa chất lượng dữ liệu và đặc trưng. Nhờ đó, chúng ta đã chốt được pipeline tối ưu nhất ([TN3.3 - M04](file:///home/dacekey/AIL303_SUM26/paper/workspace/KietBA/phase1/m04/M04_exphase3_insights.md)):
* **Đặc trưng:** `HOG Gray + Color Histogram HSV` trên ảnh `64x64 pad_square`.
* **Giảm chiều:** `PCA 95% variance` (nén từ $1,812$ xuống $571\text{ chiều}$).
* **Cân bằng dữ liệu:** `Minority Augmentation` (tăng mẫu ảnh 2D cho các lớp hiếm).
* **Thành tích đạt được:** Val Macro F1 = **95.77%**, Test Macro F1 = **95.64%**, Tốc độ suy luận = **1.93 ms/ảnh**.

**Mục đích cốt lõi của Nhiệm vụ M05:**
Khi bước sang **Phase 2 (Mở rộng luận văn & Nghiên cứu sâu - Ablation Study)**, chúng ta cần hoàn thiện mảnh ghép khoa học cuối cùng: **Kiểm chứng thực nghiệm chéo (Cross-model Benchmarking)**. 
Bạn sẽ giữ nguyên bộ dữ liệu và ma trận đặc trưng chiến thắng vàng từ Phase 1 ($571\text{ chiều}$ sau PCA), sau đó thử nghiệm chạy đối chứng với các đại diện tiêu biểu nhất của các họ thuật toán Học máy (Tree Bagging, Gradient Boosting, Linear Models, Distance-based, Neural Net cổ điển).

Việc này giúp bài luận văn/bài báo khoa học của nhóm trả lời triệt để câu hỏi của hội đồng phản biện: *"Liệu kết quả xuất sắc đạt được có phải chỉ nhờ mô hình SVM, hay bộ đặc trưng của nhóm khi ghép vào các thuật toán ML khác vẫn cho hiệu năng tốt? Và bản chất toán học nào khiến thuật toán này thắng thuật toán kia trên dữ liệu biển báo?"*

---

## 2. NGUỒN DỮ LIỆU KẾ THỪA TỪ PHASE 1

Bạn **tuyệt đối không cần (và không được) trích xuất lại ảnh từ đầu** để tiết kiệm thời gian và đảm bảo hệ quy chiếu đối chứng chính xác 100%. Nạp trực tiếp ma trận đặc trưng đã lưu sẵn trong cache của M04:
* **Thư mục chứa cache đặc trưng:** `paper/workspace/KietBA/phase1/m04_outputs/cache/` (hoặc file `.npz` tương ứng của thí nghiệm `TN3.3`).
* **Các biến ma trận cần nạp:**
  * `X_train_pca`: Ma trận đặc trưng tập Train sau PCA ($7,035 \times 571$).
  * `y_train`: Nhãn tập Train ($7,035$).
  * `X_val_pca`: Ma trận đặc trưng tập Validation ($824 \times 571$).
  * `y_val`: Nhãn tập Validation ($824$).
  * `X_test_pca`: Ma trận đặc trưng tập Test ($851 \times 571$).
* **Tài liệu lý luận tham chiếu:** Đọc lại [LOG-02](file:///home/dacekey/AIL303_SUM26/paper/vault/log/02_26-06-29_ml_model_selection_and_locking.md) để nắm cơ sở lý thuyết ban đầu khi nhóm khóa mô hình SVM.

---

## 3. THIẾT LẬP CẤU HÌNH BAN ĐẦU CHO CÁC MÔ HÌNH (LÝ DO KHOA HỌC)

Trong nhiệm vụ M05 này, **chưa cần thực hiện Fine-tuning hay Grid Search siêu tham số cho từng model**. Mục tiêu là đánh giá năng lực nội tại (intrinsic baseline capacity) của từng họ thuật toán với cấu hình chuẩn mực ban đầu (Standard/Default Config có bật xử lý mất cân bằng lớp).

Dưới đây là danh sách 6 mô hình thí nghiệm và cấu hình thiết lập bắt buộc:

### 3.1. Mô hình 1: Support Vector Machine - SVC RBF *(Hệ quy chiếu vàng / Gold Standard)*
* **Thư viện:** `sklearn.svm.SVC`
* **Cấu hình:** `SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', random_state=42)`
* **Lý do thiết lập:** Đây là mô hình chiến thắng từ Phase 1, giữ vai trò làm mốc chuẩn (Benchmark Baseline) để so sánh với 5 mô hình mới.

### 3.2. Mô hình 2: Random Forest Classifier *(Đại diện Bagging Ensemble Trees)*
* **Thư viện:** `sklearn.ensemble.RandomForestClassifier`
* **Cấu hình:** `RandomForestClassifier(n_estimators=200, max_depth=None, class_weight='balanced', n_jobs=-1, random_state=42)`
* **Lý do thiết lập:** Sử dụng `200` cây quyết định (đủ lớn để luật số lớn phát huy tác dụng, triệt tiêu phương sai variance mà không làm chậm quá mức). `class_weight='balanced'` tự động điều chỉnh trọng số bootstrap theo tỷ lệ nghịch với tần suất lớp, giúp cây không bị bỏ rơi lớp thiểu số.

### 3.3. Mô hình 3: LightGBM Classifier *(Đại diện Gradient Boosting Trees)*
* **Thư viện:** `lightgbm.LGBMClassifier` *(hoặc `xgboost.XGBClassifier` nếu không cài sẵn LightGBM)*
* **Cấu hình:** `LGBMClassifier(n_estimators=200, learning_rate=0.05, class_weight='balanced', n_jobs=-1, random_state=42, verbose=-1)`
* **Lý do thiết lập:** Boosting xây dựng cây tuần tự để sửa sai cho cây trước. Cấu hình `learning_rate=0.05` và `200` cây giữ cho mô hình học từ từ, tránh hiện tượng Overfitting quá nhanh khi gặp không gian đặc trưng mật độ cao $571\text{ chiều}$.

### 3.4. Mô hình 4: Logistic Regression *(Đại diện Linear Probabilistic Model)*
* **Thư viện:** `sklearn.linear_model.LogisticRegression`
* **Cấu hình:** `LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs', random_state=42)`
* **Lý do thiết lập:** Thuật toán tối ưu hóa xác suất tuyến tính bằng hàm Log-loss. Thiết lập `max_iter=1000` đảm bảo bộ giải `lbfgs` hội tụ hoàn toàn trên bài toán phân loại đa lớp (46 classes). `C=1.0` là chuẩn chuẩn hóa L2 tiêu chuẩn.

### 3.5. Mô hình 5: K-Nearest Neighbors - KNN *(Đại diện Instance/Distance-based Model)*
* **Thư viện:** `sklearn.neighbors.KNeighborsClassifier`
* **Cấu hình:** `KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=2, n_jobs=-1)`
* **Lý do thiết lập:** Chọn `k=5` lân cận là mức cân bằng giữa chống nhiễu (k=1) và làm mờ ranh giới (k quá lớn). Tham số `weights='distance'` gán trọng số tầm ảnh hưởng tỷ lệ nghịch với khoảng cách Euclid (`p=2`), giúp các điểm lân cận gần sát có tiếng nói quyết định mạnh hơn trong không gian $571\text{ chiều}$.

### 3.6. Mô hình 6: Multi-layer Perceptron - MLP *(Đại diện Classical Neural Network)*
* **Thư viện:** `sklearn.neural_network.MLPClassifier`
* **Cấu hình:** `MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', solver='adam', max_iter=500, early_stopping=True, random_state=42)`
* **Lý do thiết lập:** Kiến trúc 2 lớp ẩn `(256, 128)` là mô hình mạng nơ-ron truyền thống vừa đủ sức mạnh ánh xạ phi tuyến từ $571$ đầu vào ra $46$ lớp đầu ra. Bật `early_stopping=True` (tự động tách 10% train làm validation nội bộ) là cực kỳ quan trọng để ngăn chặn mạng MLP bị học thuộc lòng (Memorization / Overfitting) trên tập dữ liệu HDLSS.

---

## 4. HƯỚNG DẪN TRIỂN KHAI IMPLEMENTATION TỪNG BƯỚC

Bạn hãy tạo một notebook mới tại workspace mang tên:
`paper/workspace/KietBA/phase2/M05_ml_models_benchmarking.ipynb` *(Hoặc thư mục workspace của bạn)*.

### Bước 1: Nạp Dữ liệu & Thiết lập Ma trận Thí nghiệm
* Đọc các mảng `X_train_pca`, `y_train`, `X_val_pca`, `y_val`, `X_test_pca`, `y_test` từ file cache Phase 1.
* Khởi tạo dictionary chứa 6 mô hình với cấu hình chuẩn ở Mục 3.

### Bước 2: Viết Vòng lặp Huấn luyện & Đo lường Tự động
Khớp (`fit`) từng mô hình trên tập Train và đo lường nghiêm ngặt 5 chỉ số sau:
1. **Thời gian Huấn luyện (`train_time_sec`):** Đo bằng `time.perf_counter()`.
2. **Tốc độ Suy luận (`inference_ms_per_image`):** Đo thời gian `predict` trên toàn bộ tập Val/Test chia cho tổng số ảnh, đổi ra mili-giây.
3. **Độ chính xác (`val_accuracy` / `test_accuracy`).**
4. **Macro F1-score (`val_macro_f1` / `test_macro_f1`).**
5. **Recall trung bình trên 10 Lớp Thiểu số (`minority_macro_recall`):** Lọc riêng F1/Recall của 10 lớp thiểu số chốt từ M01 (`R.301e`, `W.205c`, `I.409`, `S.505a_Xe máy`, `W.205a`, `W.225`...).

### Bước 3: Xuất Bảng Số liệu Tổng hợp
Lưu toàn bộ kết quả vào file CSV tại: `paper/workspace/KietBA/phase2/outputs/M05_ml_models_comparison_results.csv` theo bảng mẫu:

| Mã TN | Họ Thuật toán | Tên Mô hình | Val Accuracy (%) | Val Macro F1 (%) | Minority Recall (%) | Train Time (s) | Inference (ms/ảnh) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TN4.1** | Support Vector Machine | `SVC RBF (Gold Standard)` | *96.72%* | *95.77%* | *...* | *...* | *~2.55* |
| **TN4.2** | Bagging Trees | `Random Forest (200 trees)` | *...* | *...* | *...* | *...* | *...* |
| **TN4.3** | Boosting Trees | `LightGBM (200 trees)` | *...* | *...* | *...* | *...* | *...* |
| **TN4.4** | Linear Model | `Logistic Regression` | *...* | *...* | *...* | *...* | *...* |
| **TN4.5** | Distance-based | `K-Nearest Neighbors (k=5)` | *...* | *...* | *...* | *...* | *...* |
| **TN4.6** | Neural Network | `MLP (256-128 ReLU)` | *...* | *...* | *...* | *...* | *...* |

---

## 5. YÊU CẦU TRỰC QUAN HÓA BẰNG CHỨNG (VISUALIZATION DELIVERABLES)

Bên cạnh bảng số liệu CSV, bạn bắt buộc phải viết code vẽ và lưu lại **ít nhất 2 biểu đồ trực quan hóa có giá trị cao** (lưu định dạng `.png` độ phân giải cao 300 DPI vào thư mục `outputs/figures/`):

### 5.1. Biểu đồ Trade-off giữa Chất lượng Nhận diện & Tốc độ (`M05_f1_vs_inference_speed.png`)
* **Loại biểu đồ:** Scatter Plot (hoặc Bubble Chart).
* **Trục hoành (X-axis):** Tốc độ suy luận `Inference Speed (ms/image)` *(quy mô log scale nếu chênh lệch lớn)*.
* **Trục tung (Y-axis):** Điểm `Val Macro F1 (%)`.
* **Màu sắc/Kích thước bọt:** Tương ứng với thời gian huấn luyện hoặc nhóm thuật toán.
* **Ý nghĩa:** Trực quan hóa rõ ràng góc phần tư "Sweet Spot" (nhanh nhất + chính xác nhất) để chứng minh mô hình nào khả thi nhất cho triển khai thực tế trên xe tự lái.

### 5.2. Biểu đồ Radar/Bar Chart Đối chứng Hiệu năng Lớp Thiểu số (`M05_minority_classes_f1_comparison.png`)
* **Loại biểu đồ:** Grouped Bar Chart hoặc Radar Chart.
* **Nội dung:** So sánh điểm F1 từng lớp trong **10 lớp biển báo thiểu số** giữa 3 mô hình đại diện mạnh nhất (Ví dụ: `SVC RBF` vs `Random Forest` vs `MLP`).
* **Ý nghĩa:** Chỉ ra minh chứng trực quan rằng thuật toán cây quyết định hay KNN bị "gãy" (Recall tụt về 0%) ở những biển báo hiếm gặp nào so với SVM.

---

## 6. YÊU CẦU PHÂN TÍCH BẢN CHẤT TOÁN HỌC / HÌNH HỌC (THEORETICAL ABLATION ANALYSIS)

Phần quan trọng nhất của M05 không chỉ là con số, mà là **phần viết báo cáo phân tích nhận xét trong markdown notebook**. Bạn phải trả lời được các câu hỏi bản chất sau dựa trên số liệu thực đo:

### 6.1. Vì sao SVM RBF vẫn giữ vị thế thống trị (Hoặc nhỉnh hơn rõ rệt)?
* **Giải thích bản chất:** Nhắc lại cơ chế **Support Vectors**. Trong không gian $571\text{ chiều}$ liên tục của gradient HOG, SVM tìm siêu phẳng tối ưu lề tối đa chỉ dựa vào các điểm ranh giới. Thủ thuật kernel RBF chiếu dữ liệu lên không gian vô hạn giúp ôm sát các biến dạng hình học phi tuyến mà không bị Overfitting.

### 6.2. Vì sao nhóm Cây quyết định (Random Forest / LightGBM) gặp hạn chế?
* **Giải thích bản chất:** Các thuật toán cây phân chia không gian bằng các mặt phẳng **vuông góc với trục tọa độ (Axis-aligned orthogonal splits)**. Tuy nhiên, các vector HOG nén qua PCA là sự tổ hợp tuyến tính liên tục của các góc đường chéo. Để cắt một đường chéo phi tuyến bằng các lát cắt vuông góc, cây phải phân nhánh thành vô số bậc thang nhỏ vụn $\rightarrow$ Gây ra hiện tượng **phân mảnh vùng quyết định (Decision Boundary Fragmentation)**, dẫn đến Overfit trên Train và suy luận chậm do phải duyệt qua hàng trăm tầng cây.

### 6.3. Vì sao K-Nearest Neighbors (KNN) sa sút trong không gian này?
* **Giải thích bản chất:** Dù đã giảm chiều còn $571$, đây vẫn là số chiều rất lớn đối với thuật toán tính khoảng cách. Theo định lý **Lời nguyền cô đặc khoảng cách (Distance Concentration Curse)** trong không gian nhiều chiều, khoảng cách Euclid từ một điểm truy vấn đến điểm gần nhất ($D_{\min}$) và đến điểm xa nhất ($D_{\max}$) có xu hướng tiến tới bằng nhau ($\frac{D_{\max} - D_{\min}}{D_{\min}} \rightarrow 0$). KNN bị mất phương hướng khi bầu chọn lân cận.

### 6.4. Vì sao Logistic Regression bị giới hạn siêu phẳng?
* **Giải thích bản chất:** Logistic Regression là hàm phân loại tuyến tính toàn cục ($w^T x + b = 0$). Ranh giới siêu phẳng cứng nhắc không thể phân tách các cụm biển báo bị xô lệch góc chụp 3D ngoài trời (vốn có hình dạng cụm lồi lõm phức tạp).

---

## 7. TIÊU CHÍ HOÀN THÀNH & BÀN GIAO (DONE CRITERIA)

Nhiệm vụ M05 được đánh giá hoàn thành 100% khi thành viên phụ trách hoàn tất:
- [x] Notebook `M05_ml_models_benchmarking.ipynb` chạy hoàn tất từ đầu đến cuối không lỗi.
- [x] File CSV `M05_ml_models_comparison_results.csv` chứa đầy đủ 6 mô hình.
- [x] Ít nhất 2 biểu đồ trực quan hóa `.png` trong `outputs/figures/`.
- [x] Bài phân tích bản chất toán học/hình học giải thích số liệu được viết chỉn chu ngay trong Notebook hoặc file báo cáo Markdown `M05_insights.md`.

**Chúc bạn thực thi nhiệm vụ thuận lợi và mang lại minh chứng khoa học tuyệt đối cho luận văn!** 🚀
