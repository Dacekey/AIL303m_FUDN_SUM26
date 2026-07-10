# NHIỆM VỤ M05B: Chuẩn hóa & Hoàn thiện Thí nghiệm Đối chứng Tối ưu Các Mô hình Học máy (Tuned ML Benchmarking Standardization & Ablation Supplementary)

- **Mã nhiệm vụ:** `M05B` *(Nhiệm vụ nâng cấp & chuẩn hóa kế thừa từ M05)*
- **Ngày ban hành:** 10/07/2026
- **Người thực hiện phụ trách:** Thành viên 4 (TruongDT - ML & Pipeline Engineer)
- **Giai đoạn thực hiện:** Giai đoạn Mở rộng / Thực nghiệm Đối chứng (Phase 2)
- **Đầu ra bắt buộc (Deliverables Chuẩn hóa):**
  1. Bảng số liệu đối chứng sau tinh chỉnh (Tuned-Benchmarking) đã được chuẩn hóa đơn vị & chỉ số (`M05_ml_models_comparison_results.csv`).
  2. Biểu đồ Trade-off được chuẩn hóa trục tốc độ (`M05_f1_vs_inference_speed.png`).
  3. Biểu đồ trực quan hóa đối chứng hiệu năng trên 10 Lớp thiểu số (`M05_minority_classes_f1_comparison.png`).
  4. Cấu trúc thư mục bàn giao chuẩn mực tại `paper/workspace/TruongDT/phase2_new/outputs/`.

---

## PHẦN 1: GHI NHẬN & BIỂU DƯƠNG NHỮNG NỖ LỰC XUẤT SẮC CỦA THÀNH VIÊN (SCIENTIFIC BREAKTHROUGHS RECOGNITION)

Ban chủ nhiệm dự án cùng nhóm nghiên cứu **chính thức ghi nhận, đánh giá cực kỳ cao và biểu dương sự chủ động vượt mức yêu cầu** của Thành viên phụ trách trong quá trình thực thi Nhiệm vụ M05 ban đầu. 

Mặc dù đặc tả gốc M05 (`M05_26-07-03_ml_models_benchmarking_and_ablation.md`) chỉ yêu cầu khảo sát năng lực nội tại ở cấu hình chuẩn mực ban đầu (Standard Baseline) để giảm tải khối lượng tính toán, bạn đã không quản ngại tài nguyên máy tính và thời gian để triển khai thêm khâu **Khảo sát lưới siêu tham số (GridSearchCV / Hyperparameter Tuning) cho toàn bộ 6 họ mô hình ML**.

Sự nỗ lực vượt bậc này đã mang lại **giá trị khoa học to lớn và bước tiến học thuật vượt cấp** cho luận văn/bài báo của nhóm:

### 1.1. Minh chứng Công bằng Tuyệt đối (Fair Tuned-Benchmarking)
Lỗ hổng lớn nhất khi phản biện một bài báo so sánh mô hình chính là sự thiên lệch tham số (Parameter Bias): *Liệu SVM thắng có phải nhờ được chọn cấu hình tốt từ trước, còn các thuật toán khác thua do chạy cấu hình mặc định?* 

Kết quả tuning toàn diện của bạn trong `M05_ml_models_benchmarking.ipynb` đã mang lại **lời đáp trả sắt đá tuyệt đối**:
* Ngay cả khi được cho cơ hội dò tìm siêu tham số tối ưu nhất trên không gian đặc trưng PCA 571 chiều ($X_{\text{train\_pca}}$), nhóm thuật toán Cây quyết định và Lân cận gần nhất vẫn **thua kém rõ rệt** so với bộ phân loại SVM vàng:
  - **Random Forest (Tuned):** Macro F1 = **84.74%**
  - **LightGBM (Tuned):** Macro F1 = **86.70%** *(thậm chí khi tuning quá mức còn bị overfit tụt xuống 79.93%)*
  - **K-Nearest Neighbors - KNN (Tuned):** Macro F1 = **94.23%**
  - **Multi-layer Perceptron - MLP (Tuned):** Macro F1 = **93.62%**
  - **Support Vector Classifier - SVC RBF (Tuned):** Macro F1 = **95.69%** (và `Logistic Regression C=0.1` đạt **95.42%**)

### 1.2. Khối Lý luận Toán học & Hình học Khóa chặt Luận điểm (`M05_insights.md`)
Bài báo cáo Markdown `M05_insights.md` của bạn là một **tài sản nghiên cứu xuất sắc và vô giá**. Bạn đã giải thích vô cùng chính xác, sắc bén bản chất hình học đằng sau các con số:
* **Sự bất tương thích hình học của Tree-based:** Các vector HOG sau nén PCA là sự tổ hợp tuyến tính phân bố theo các góc đường chéo liên tục. Khi thuật toán cây quyết định buộc phải chia cắt không gian bằng các mặt phẳng **trực giao vuông góc với trục tọa độ (Axis-aligned orthogonal splits)**, ranh giới phân loại bị phân nhánh thành vô số lát cắt bậc thang vụn vặt $\rightarrow$ Gây ra hiện tượng **Phân mảnh vùng quyết định (Decision Boundary Fragmentation)**, dẫn đến Overfitting trên Train và F1 thấp trên Test.
* **Lời nguyền cô đặc khoảng cách (Distance Concentration Curse):** Giải thích thấu đáo vì sao trong không gian $\mathbb{R}^{571}$, tỷ số $\frac{D_{\max} - D_{\min}}{D_{\min}} \rightarrow 0$ khiến cơ chế bầu chọn lân cận của KNN bị mất phương hướng.
* **Giới hạn siêu phẳng cứng nhắc của Logistic Regression:** Giải thích rõ lý do hàm tuyến tính toàn cục $(\mathbf{w}_k - \mathbf{w}_j)^T \mathbf{x} + (b_k - b_j) = 0$ không thể uốn lượn ôm sát các cụm biển báo bị xô lệch góc chụp 3D phức tạp ngoài trời như kernel RBF phi tuyến.

👉 **KẾT LUẬN CHIẾN LƯỢC PHẦN 1:** Toàn bộ hướng đi mới và khối lý luận toán học của bạn sẽ được **chính thức giữ lại, khóa cố định và tích hợp thẳng vào bài luận văn Phase 2** như một minh chứng cho sự công phu, triệt để và tính đúng đắn bản chất toán học của Pipeline Phase 1-New!

---

## PHẦN 2: YÊU CẦU CHUẨN HÓA & CÁC NHIỆM VỤ BỔ SUNG HOÀN THIỆN (STANDARDIZATION TASKS)

Để bước tiến khoa học xuất sắc kể trên đạt **điểm 10/10 hoàn hảo trước Hội đồng bảo vệ** và không bị trừ điểm oan do hiểu lầm về mặt trình bày số liệu hay quy chuẩn bàn giao, bạn được giao thực hiện **Nhiệm vụ M05B (Chuẩn hóa & Hoàn thiện M05 Tuned-Benchmarking)** với 4 hạng mục kỹ thuật nhanh dưới đây:

### Hạng mục 1: Chuẩn hóa lại Đơn vị Tốc độ Suy luận (`Inference Speed`)
* **Vấn đề tồn đọng:** Hiện tại, code đang đo tổng thời gian chạy `model.predict(X_test_pca)` trên toàn bộ $851$ ảnh tập Test rồi ghi thẳng con số ra đơn vị **giây (s)** vào dataframe (`1.6867 s` cho SVC RBF, `0.0262 s` cho Random Forest...). Khi Hội đồng nhìn vào bảng số liệu thấy Inference Time là `1.68` hay `43.7` sẽ bị hiểu lầm tai hại là mô hình mất 1.68 giây cho 1 ảnh (cực kỳ chậm, không đủ điều kiện thời gian thực trên xe tự lái).
* **Yêu cầu chuẩn hóa:**
  1. Trong notebook `M05_ml_models_benchmarking.ipynb`, sửa lại công thức tính cột tốc độ suy luận thành đơn vị **mili-giây trên ảnh (`ms/image` / `ms/ảnh`)**:
     $$\text{Inference Speed (ms/image)} = \frac{\text{infer\_time\_seconds}}{N_{\text{test\_images}}} \times 1000 = \frac{\text{infer\_time}}{851} \times 1000$$
  2. Cập nhật lại tên cột trong bảng kết quả thành: `Inference (ms/ảnh)` (hoặc `Inference Speed (ms/image)`).
  3. *Kết quả chuẩn minh họa:* SVC RBF sẽ đạt $\frac{1.6867}{851} \times 1000 \approx \mathbf{1.98\text{ ms/ảnh}}$ (hoàn toàn đáp ứng realtime edge AI).

### Hạng mục 2: Chuẩn hóa lại Chỉ số Recall trên 10 Lớp Thiểu số (`Minority Macro Recall`)
* **Vấn đề tồn đọng:** Hiện tại code tính Recall thiểu số bằng lệnh:
  ```python
  recalls_per_class = recall_score(y_test, y_pred, average=None)
  minority_recall = np.min(recalls_per_class)
  ```
  Lệnh `np.min()` sẽ lấy giá trị Recall của **lớp đơn lẻ có thành tích thấp nhất trong toàn bộ 46 lớp** (ra các phân số đúng như $0.3333 = 1/3$, $0.1905 = 4/21$, $0.5000 = 1/2$ do có lớp chỉ có 2-3 mẫu). Điều này làm bóp méo và đánh tụt chỉ số, không phản ánh đúng năng lực nhận diện của mô hình trên các lớp biển báo hiếm.
* **Yêu cầu chuẩn hóa:**
  1. Khai báo chính xác danh sách nhãn của **10 lớp thiểu số chốt từ Phase 1** (các biển có số lượng ảnh mẫu thấp hoặc bị nhầm lẫn nhiều):
     ```python
     # Danh sách 10 lớp thiểu số tham chiếu chuẩn của dự án
     minority_classes = ['R.301e', 'W.205c', 'I.409', 'S.505a_Xe máy', 'W.205a', 'W.225', 'P.124d', 'W.203c', 'P.103a', 'W.239b_']
     # (Hoặc lọc các lớp có support trong y_train < ngưỡng thiểu số chốt từ M01)
     ```
  2. Lọc riêng mảng điểm Recall (hoặc Macro F1) của đúng 10 lớp thiểu số này và tính **trung bình cộng (`np.mean()`)**:
     ```python
     # Giả sử class_names là mảng tên các lớp từ LabelEncoder
     minority_indices = [i for i, cls in enumerate(class_names) if cls in minority_classes]
     minority_macro_recall = np.mean(recalls_per_class[minority_indices])
     ```
  3. Cập nhật cột vào dataframe với tên chuẩn: `Minority Recall (%)` (hoặc `Minority Macro F1 (%)`).

### Hạng mục 3: Bổ sung Biểu đồ Đối chứng Hiệu năng 10 Lớp Thiểu số (`M05_minority_classes_f1_comparison.png`)
* **Vấn đề tồn đọng:** Bạn đã vẽ biểu đồ Trade-off và biểu đồ `Comparison of Tuned Macro F1 Scores` (rất tuyệt vời), nhưng để hoàn thiện mảnh ghép chứng cứ theo Mục 5.2 của M05 gốc, chúng ta cần trực quan hóa cụ thể hiệu năng trên 10 lớp thiểu số.
* **Yêu cầu bổ sung:**
  1. Viết code vẽ và xuất ra file hình ảnh độ phân giải cao (`300 DPI`) mang tên: `M05_minority_classes_f1_comparison.png`.
  2. **Loại biểu đồ:** Grouped Bar Chart (hoặc Radar Chart).
  3. **Trục hoành (X-axis):** Tên của 10 lớp biển báo thiểu số (`R.301e`, `W.205c`...).
  4. **Trục tung (Y-axis):** Điểm F1-score (hoặc Recall) của từng lớp.
  5. **Các nhóm cột đối chứng:** Vẽ đối chứng giữa ít nhất 3 mô hình đại diện sau tuning: **`SVC RBF (Tuned)` vs. `Random Forest (Tuned)` vs. `MLP (Tuned)`**.
  6. *Ý nghĩa minh chứng:* Chỉ ra minh chứng trực quan sắt đá rằng các mô hình cây quyết định hay MLP dù đã tuning vẫn bị F1/Recall = 0% ở những biển báo góc cạnh bị bóp méo phối cảnh cực đoan, trong khi SVM RBF giữ được độ suy rộng cực tốt.

### Hạng mục 4: Chuẩn hóa Trục Trade-off & Cấu trúc Thư mục Lưu trữ (`outputs/`)
* **Vấn đề tồn đọng:** Toàn bộ các file `.csv`, `.png` và `.json` đang bị lưu thả trôi nổi tại thư mục gốc `paper/workspace/TruongDT/phase2_new/`. Biểu đồ Trade-off đang dùng trục hoành là `Inference Time (s)`.
* **Yêu cầu chuẩn hóa:**
  1. Tạo đúng cấu trúc thư mục quy chuẩn của dự án:
     ```
     paper/workspace/TruongDT/phase2_new/outputs/
     ├── M05_ml_models_comparison_results.csv   (File CSV đã chuẩn hóa đơn vị & recall)
     ├── best_params.json                       (File JSON lưu tham số tuning)
     └── figures/
         ├── M05_f1_vs_inference_speed.png          (Biểu đồ Trade-off chuẩn hóa trục X ra ms/ảnh)
         ├── M05_minority_classes_f1_comparison.png (Biểu đồ bổ sung 10 lớp thiểu số)
         ├── Comparison_of_Tuned_Macro_F1_Scores.png (Biểu đồ bar chart tuning của bạn - giữ lại)
         └── Confusion_Matrix_SVC_RBF.png           (Ma trận nhầm lẫn của bạn - giữ lại)
     ```
  2. Vẽ lại biểu đồ Trade-off (`M05_f1_vs_inference_speed.png`) với trục hoành là `Inference Speed (ms/image)` (tốc độ mili-giây trên ảnh) thay vì giây, bảo đảm góc phần tư "Sweet Spot" (nhanh nhất + chính xác nhất) hiển thị chính xác SVC RBF và Logistic Regression.
  3. Trong code notebook, bảo đảm mọi lệnh `plt.savefig()` và `to_csv()` đều trỏ về đúng các đường dẫn tương đối `outputs/` và `outputs/figures/`.

---

## PHẦN 3: TIÊU CHÍ HOÀN THÀNH & BÀN GIAO M05B (DONE CRITERIA)

Nhiệm vụ M05B được đánh giá hoàn thành **100% trọn vẹn và đạt điểm 10 tuyệt đối** khi thành viên phụ trách (TruongDT) kiểm tra hoàn tất các mục sau:
- [ ] Notebook `M05_ml_models_benchmarking.ipynb` đã được cập nhật code chuẩn hóa đơn vị tốc độ `ms/ảnh` và cách tính trung bình `Minority Recall` trên 10 lớp.
- [ ] Thư mục `paper/workspace/TruongDT/phase2_new/outputs/` được tạo mới và chứa file `M05_ml_models_comparison_results.csv` với số liệu chính xác 100%.
- [ ] Thư mục `paper/workspace/TruongDT/phase2_new/outputs/figures/` chứa đầy đủ 4 biểu đồ độ phân giải cao (`300 DPI`), trong đó có file bổ sung `M05_minority_classes_f1_comparison.png` và file chuẩn hóa `M05_f1_vs_inference_speed.png`.
- [ ] File `M05_insights.md` được giữ nguyên (hoặc bổ sung thêm 1-2 câu trích dẫn hình ảnh từ `M05_minority_classes_f1_comparison.png` nếu muốn làm rõ hơn).

**Chúc bạn thực hiện chuẩn hóa nhanh chóng và đóng gói thành công một mảnh ghép thực nghiệm khoa học đẳng cấp cao cho Phase 2!** 🎯🔥
