# NHIỆM VỤ M06: Thực thi Tinh chỉnh & Tối ưu hóa Siêu tham số Mô hình Support Vector Machine (SVM Fine-tuning & Hyperparameter Optimization)

- **Mã nhiệm vụ:** `M06`
- **Ngày ban hành:** 05/07/2026
- **Người thực hiện phụ trách:** Thành viên 4 (ML & Pipeline Engineer) *(Nhiệm vụ độc lập chuyên sâu)*
- **Giai đoạn thực hiện:** Giai đoạn Mở rộng / Thực nghiệm Đối chứng (Phase 2)
- **Đầu ra bắt buộc (Deliverables):**
  1. Bảng số liệu khảo sát siêu tham số & đối chứng Phase 1-New vs. Phase 2-New (`M06_svm_finetuning_results.csv`).
  2. Mô hình SVM tối ưu nhất sau Fine-tuning (`M06_best_tuned_svm_model.joblib`).
  3. Biểu đồ Heatmap Grid Search ($C$ vs. $\gamma$) và ma trận nhầm lẫn cuối cùng trên tập Test.
  4. Báo cáo phân tích chuyên sâu giải thích bản chất toán học/hình học của siêu phẳng tối ưu (`M06_svm_finetuning_insights.md`).

---

## 1. MỤC ĐÍCH & TẦM QUAN TRỌNG CỦA NHIỆM VỤ (WHY FINE-TUNE SVM NOW?)

Tại **Phase 1 (Giai đoạn Thực nghiệm Tối ưu hóa Pipeline - M01 đến M04)**, tuân thủ nghiêm ngặt quyết định chiến lược tại [LOG-02](file:///home/dacekey/AIL303_SUM26/paper/vault/log/02_26-06-29_ml_model_selection_and_locking.md), nhóm nghiên cứu đã **khóa cố định bộ phân loại ở tiêu chuẩn vàng: `StandardScaler() + SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced')`**. Việc khóa thuật toán này đã giúp chúng ta cô lập hoàn toàn các biến số mô hình, tập trung 100% nguồn lực vào việc tinh chế chất lượng dữ liệu và kỹ thuật đặc trưng.

Kết quả tổng kết tại báo cáo [R01](file:///home/dacekey/AIL303_SUM26/paper/result/report_new/R01_26.07.05_report-phase1-new.md) đã chứng minh thành tựu rực rỡ của chiến lược này:
* **Bộ tham số tối ưu chốt tại Phase 1-New:** Đặc trưng `HOG Gray + Color Hist HSV` trên ảnh `64x64 pad_square`, nén giảm chiều **PCA 95% variance** (từ 1,812 xuống **571 chiều**), kết hợp tăng cường dữ liệu **Minority Augmentation** trên 10 lớp thiểu số.
* **Thành tích Phase 1-New:** Val Macro F1 = **95.77%**, Test Macro F1 = **95.64%**, Test Accuracy = **96.24%**, Tốc độ suy luận siêu nhanh = **1.93 ms/ảnh**.

**Tại sao phải thực hiện Nhiệm vụ M06 tại Phase 2?**
Cấu hình siêu phẳng $C=10.0$ và $\gamma=\text{'scale'}$ được sử dụng trong suốt Phase 1 chỉ là **cấu hình mặc định nhắm chừng (sensible default)** được thiết lập cho không gian đặc trưng gốc nghìn chiều chưa qua xử lý. 

Khi bước sang **Phase 2**, không gian đặc trưng đã được biến đổi hoàn toàn:
1. **Sự thay đổi về hình học không gian:** Ma trận đặc trưng đã được PCA nén nhỏ từ $1,812\text{ chiều}$ xuống còn $571\text{ chiều}$, loại bỏ hơn $68\%$ ma trận nhiễu nền. Mật độ phân bố của các cụm class trong không gian $571\text{ chiều}$ trở nên đặc chắc và phân cực tốt hơn rất nhiều so với không gian ban đầu.
2. **Sự thay đổi về ranh giới quyết định:** Khi độ nhiễu giảm đi, lề siêu phẳng tối ưu (controlled by regularization parameter $C$) và độ rộng vùng ảnh hưởng của mỗi vector hỗ trợ (controlled by kernel bandwidth $\gamma$) không còn giống như trước. Thậm chí, loại hạt nhân phi tuyến (RBF) chưa chắc đã ưu việt hơn hạt nhân Đa thức (Polynomial Kernel) trong việc mô hình hóa tương tác góc cạnh và màu sắc.

**Sứ mệnh của M06:** Bạn được **mở khóa hoàn toàn mô hình SVM**. Nhiệm vụ của bạn là sử dụng bộ tham số dữ liệu vàng từ R01, tiến hành khảo sát và tinh chỉnh tối ưu hóa siêu tham số (Hyperparameter Optimization / Fine-tuning) để tìm ra giới hạn hiệu năng tuyệt đối của thuật toán chiến thắng. Khi kết hợp cùng nhiệm vụ đối chứng 6 họ mô hình ML ([M05](file:///home/dacekey/AIL303_SUM26/paper/mission/phase2_new/M05_26-07-03_ml_models_benchmarking_and_ablation.md)), bạn sẽ hoàn thiện bức tranh khoa học toàn diện cho luận văn/báo cáo của nhóm!

---

## 2. NGUỒN DỮ LIỆU & TÀI LIỆU THAM CHIẾU

Để đảm bảo tính nhất quán khoa học tuyệt đối và tiết kiệm thời gian tính toán, bạn **tuyệt đối không trích xuất lại đặc trưng hay chạy lại tiền xử lý từ đầu**. Hãy nạp trực tiếp ma trận đặc trưng chiến thắng đã lưu trong cache từ nhiệm vụ M04.

* **Thư mục chứa ma trận cache:** [paper/workspace/KietBA/phase1_new/m04_outputs/cache/](file:///home/dacekey/AIL303_SUM26/paper/workspace/KietBA/phase1_new/m04_outputs/cache)
* **Các biến ma trận cần nạp (file `.npz` của cấu hình TN3.3):**
  - `X_train_pca`: Ma trận đặc trưng tập Train sau PCA ($7,035 \times 571$).
  - `y_train`: Nhãn tập Train ($7,035$).
  - `X_val_pca`: Ma trận đặc trưng tập Validation ($824 \times 571$).
  - `y_val`: Nhãn tập Validation ($824$).
  - `X_test_pca`: Ma trận đặc trưng tập Test ($851 \times 571$).
* **Tài liệu lý luận tham chiếu bắt buộc đọc trước khi code:**
  1. Báo cáo tổng kết R01: [R01_26.07.05_report-phase1-new.md](file:///home/dacekey/AIL303_SUM26/paper/result/report_new/R01_26.07.05_report-phase1-new.md) *(Nắm vững bộ tham số vàng và lỗi tồn đọng)*.
  2. Quyết định khóa/mở mô hình: [LOG-02](file:///home/dacekey/AIL303_SUM26/paper/vault/log/02_26-06-29_ml_model_selection_and_locking.md) *(Hiểu lý do khoa học tại sao bây giờ mới fine-tune)*.
  3. Phân tích chi tiết M04: [M04_exphase3_insights.md](file:///home/dacekey/AIL303_SUM26/paper/workspace/KietBA/phase1_new/m04/M04_exphase3_insights.md) *(Xem baseline TN3.3 trước khi fine-tune)*.
  4. Nhiệm vụ đối chứng M05: [M05_26-07-03_ml_models_benchmarking_and_ablation.md](file:///home/dacekey/AIL303_SUM26/paper/mission/phase2_new/M05_26-07-03_ml_models_benchmarking_and_ablation.md) *(Hiểu sự liên kết giữa M05 và M06)*.

---

## 3. HƯỚNG DẪN TRIỂN KHAI THỰC NGHIỆM CHI TIẾT TỪNG BƯỚC

Bạn hãy tạo một notebook mới tại workspace mang tên:
`paper/workspace/KietBA/phase2_new/M06_svm_finetuning_experiments.ipynb` *(hoặc thư mục workspace tương ứng của bạn)*.

### Bước 1: Nạp Dữ liệu Cache & Thiết lập Benchmark Baseline
* Đọc các mảng `X_train_pca`, `y_train`, `X_val_pca`, `y_val`, `X_test_pca`, `y_test` từ file cache của M04.
* Khởi tạo mô hình Baseline của Phase 1: `SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', random_state=42)`.
* Đo lường điểm `Val Macro F1`, `Val Accuracy` và `Inference Speed` làm mốc gốc để đo mức độ cải thiện của các thí nghiệm sau.

### Bước 2: Thí nghiệm 1 - Khảo sát Hạt nhân phi tuyến & tuyến tính (Kernel Exploration)
Trong không gian $571\text{ chiều}$ sau PCA, cấu trúc dữ liệu có thể phản ứng khác nhau với các hàm ánh xạ hạt nhân. Hãy huấn luyện và đánh giá trên tập Val 4 cấu hình kernel:
1. **`Linear Kernel`:** `SVC(kernel='linear', C=10.0, class_weight='balanced')` — Kiểm tra xem không gian 571 chiều đã đủ phân cực tuyến tính chưa.
2. **`RBF Kernel`:** *(Baseline M04)* — Hạt nhân cơ sở bán kính chuẩn.
3. **`Polynomial Kernel (Degree 2 & 3)`:** `SVC(kernel='poly', degree=2, C=10.0, class_weight='balanced')` và `degree=3` — Kiểm tra khả năng tạo đa tạp đa thức mô hình hóa tương tác giữa cạnh HOG và màu HSV.
4. **`Sigmoid Kernel`:** `SVC(kernel='sigmoid', C=10.0, class_weight='balanced')`.

> **Mục tiêu Bước 2:** Chọn ra loại Kernel có điểm `Val Macro F1` cao nhất để bước vào khâu tinh chỉnh lưới siêu tham số.

### Bước 3: Thí nghiệm 2 - Tối ưu hóa Lưới Siêu tham số (Grid Search / Randomized Search)
Sử dụng loại Kernel tốt nhất từ Bước 2 (dự kiến là `rbf` hoặc `poly`), thiết lập không gian tìm kiếm siêu tham số trên 2 trục cốt lõi:
* **Trục $C$ (Regularization / Margin Hardness):** `[0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]`
  - *Ý nghĩa:* $C$ nhỏ chấp nhận lề rộng hơn và có lỗi phân loại (chống overfit); $C$ lớn ép lề hẹp, phạt nặng từng điểm sai (hợp với dữ liệu sạch nhiễu).
* **Trục $\gamma$ (Kernel Bandwidth - chỉ áp dụng với RBF/Poly/Sigmoid):** `['scale', 'auto', 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]`
  - *Ý nghĩa:* $\gamma$ quyết định bán kính ảnh hưởng của mỗi support vector. $\gamma$ quá lớn dẫn đến overfit cục bộ (island boundaries); $\gamma$ quá nhỏ làm ranh giới bị phẳng hóa.

**Thực thi:** Dùng `GridSearchCV` hoặc chạy vòng lặp lồng nhau đánh giá trực tiếp trên tập `Val Split` (vì dữ liệu đã có tập Val riêng chuẩn xác từ đầu dự án). Ghi nhận ma trận kết quả để vẽ Heatmap.

### Bước 4: Thí nghiệm 3 - Khảo sát Chiến lược Trọng số Lớp (Class Weighting Strategies)
Tại báo cáo R01, chúng ta còn tồn đọng một số lớp bị nhầm lẫn nhiều trên Test như `W.205c` (F1=0%), hoặc các cặp nhầm tốc độ `P.127_60` vs `P.127_80`.
* Hãy thử nghiệm 3 cấu hình trọng số trên bộ tham số $(C^*, \gamma^*)$ tốt nhất vừa tìm được:
  1. `class_weight=None` *(Trọng số đều)*.
  2. `class_weight='balanced'` *(Tự động nghịch biến theo tần suất)*.
  3. `class_weight=custom_dict` *(Tăng trọng số phạt lên gấp 2x - 3x cho riêng các lớp bottleneck như `W.205c`, `P.127_xx`, `W.224`, `W.245a`)*.

### Bước 5: ĐÁNH GIÁ CHÍNH THỨC TRÊN TẬP TEST (MỞ KHÓA 1 LẦN DUY NHẤT)
* Lấy mô hình SVM tinh chỉnh hoàn hảo nhất (Best Tuned SVM Model) từ Bước 4.
* Tiến hành chạy `predict` trên tập **`Test Split` (851 ảnh)** đúng 1 lần duy nhất để đo lường thành tích chung cuộc.
* Xuất báo cáo `classification_report`, vẽ ma trận nhầm lẫn và lập bảng so sánh đối chứng trực tiếp giữa **Phase 1-New** và **Phase 2-New**.

---

## 4. ĐẦU RA BẮT BUỘC & QUY CHUẨN BÀN GIAO (DELIVERABLES)

Sau khi hoàn thành thực nghiệm, bạn phải nộp các tài sản sau vào thư mục output của Phase 2 (ví dụ: `paper/workspace/KietBA/phase2_new/outputs/`):

### 4.1. Bảng số liệu Khảo sát & Đối chứng Chung cuộc (`M06_svm_finetuning_results.csv`)
Bảng số liệu tổng hợp phải thể hiện rõ quá trình tiến hóa điểm số và **bảng đối chứng vàng giữa Phase 1-New và Phase 2-New** theo cấu trúc mẫu sau:

| Giai đoạn | Mô hình / Cấu hình | Kernel | $C$ | $\gamma$ | Trọng số Lớp | Test Accuracy (%) | Test Macro F1 (%) | Tốc độ Inference (ms/ảnh) | Số lượng Support Vectors | Ghi chú / Đánh giá |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Phase 1-New** | **Fixed SVM Baseline (M04)**| `rbf` | `10.0` | `scale` | `balanced` | **96.24%** | **95.64%** | **1.93 ms** | *chép từ M04* | *Hệ quy chiếu chốt tại R01* |
| **Phase 2-New** | TN1 - Linear Kernel | `linear`| `10.0` | *N/A* | `balanced` | *...* | *...* | *...* | *...* | *Khảo sát ranh giới tuyến tính* |
| **Phase 2-New** | TN1 - Polynomial Degree 2| `poly` | `10.0` | `scale` | `balanced` | *...* | *...* | *...* | *...* | *Khảo sát đa tạp đa thức* |
| **Phase 2-New** | TN2 - Grid Search Best | `???` | `???` | `???` | `balanced` | *...* | *...* | *...* | *...* | *Kết quả tối ưu lưới* |
| **🏆 Phase 2-New**| **Best Tuned SVM (Final)**| **`???`** | **`???`** | **`???`** | **`???`** | **[Kỳ vọng >96%]**| **[Kỳ vọng >96%]**| **[< 2.0 ms]**| **[???]** | **[CHIẾN THẮNG CHUNG CUỘC]** |

### 4.2. Danh sách các Artifact bắt buộc khác
1. `M06_best_tuned_svm_model.joblib`: File lưu trọng số mô hình SVM chiến thắng chung cuộc.
2. `M06_grid_search_heatmap.png`: Biểu đồ Heatmap trực quan hóa sự biến thiên của Val Macro F1 theo 2 trục $C$ và $\gamma$.
3. `M06_final_test_confusion_matrix.csv` (và `.png`): Ma trận nhầm lẫn trên tập Test của mô hình Fine-tuned.
4. `M06_svm_finetuning_insights.md`: Báo cáo tổng kết trả lời 4 câu hỏi phân tích chuyên sâu dưới đây.

---

## 5. CÁC CÂU HỎI PHÂN TÍCH CHUYÊN SÂU DÀNH CHO NGƯỜI THỰC HIỆN

Trong báo cáo tổng kết `M06_svm_finetuning_insights.md`, bạn không chỉ liệt kê con số mà phải trả lời thấu đáo 4 câu hỏi mang tính bản chất khoa học sau đây để bảo vệ trước hội đồng:

### Câu hỏi 1: Bản chất Hình học & Toán học của $C$ và $\gamma$ trong không gian PCA 571 chiều
* *Vấn đề:* Tại sao khi số chiều giảm từ 1,812 chiều gốc xuống 571 chiều (nhờ PCA), cặp siêu tham số $(C^*, \gamma^*)$ tối ưu lại bị dịch chuyển so với mức mặc định `(10.0, 'scale')`?
* *Gợi ý phân tích:* Khi loại bỏ 68.5% số chiều nhiễu, khoảng cách Euclid giữa các điểm thuộc các lớp khác nhau trở nên rõ ràng hơn (tăng Tỷ lệ Tín hiệu trên Nhiễu - SNR). Khi đó, việc tăng $C$ (áp đặt lề cứng hơn) hay giảm $\gamma$ (mở rộng bán kính siêu cầu RBF) tác động thế nào đến khả năng chống Overfitting và ranh giới quyết định? Hãy giải thích bằng hình học không gian.

### Câu hỏi 2: So sánh Hạt nhân RBF vs. Polynomial (Khả năng mô hình hóa đặc trưng thủ công)
* *Vấn đề:* Về mặt lý thuyết, hạt nhân RBF ánh xạ dữ liệu vào không gian Hilbert vô hạn chiều bằng các siêu cầu cục bộ, trong khi Polynomial Kernel tạo ra các đa tạp đa thức (polynomial manifolds) nắm bắt các tương tác nhân giữa các chiều.
* *Gợi ý phân tích:* Trong bộ đặc trưng `HOG + Color Hist`, liệu sự tương tác đa thức giữa một góc cạnh gradient (HOG) và một bin sắc độ (HSV) có mang lại lợi thế cho Kernel Poly bậc 2/3 không? Hay sự linh hoạt cục bộ của RBF vẫn chiếm ưu thế tuyệt đối trên biển báo Việt Nam? Tại sao?

### Câu hỏi 3: Khả năng Cứu vãn Lớp Bottleneck & Nhầm lẫn Tốc độ
* *Vấn đề:* Đối chiếu ma trận nhầm lẫn Test của Phase 1-New (R01) và Phase 2-New (M06). Việc tinh chỉnh siêu tham số (đặc biệt là custom class weights hoặc đổi kernel) có giúp tiêu diệt được 2 lỗi cứng đầu nhất không:
  1. Sự nhầm lẫn giữa nhóm biển giới hạn tốc độ (`P.127_60` vs. `P.127_80`).
  2. Lớp thiểu số cực đoan bị bóp méo phối cảnh `W.205c` (Giao với đường phụ - F1 Phase 1 là 0%).
* *Gợi ý phân tích:* Nếu Fine-tuning SVM vẫn không thể cứu được `W.205c` hay `P.127_60/80`, hãy chỉ ra giới hạn vật lý của biểu diễn đặc trưng (ví dụ: độ phân giải 64x64 sau khi nén PCA 571 chiều đã mất đi các nét cong vi mô của số 6 và số 8, khiến mọi siêu phẳng SVM đều bất lực). Đây sẽ là luận điểm đắt giá để kết luận luận văn!

### Câu hỏi 4: Đánh giá Trade-off Hiệu năng vs. Tốc độ & Khả năng Nhúng (Edge AI Deployment)
* *Vấn đề:* Đối với thuật toán SVM, thời gian suy luận (Inference Speed) tỷ lệ thuận với số lượng Support Vectors ($n\_support\_vectors$) được giữ lại:
  $$t_{\text{inference}} \propto n_{\text{support\_vectors}} \times d$$
* *Gợi ý phân tích:* Khi bạn tinh chỉnh $C$ và $\gamma$, tổng số lượng Support Vectors của mô hình thay đổi ra sao so với mô hình Phase 1? Nếu F1 tăng thêm $0.5\%$ nhưng số lượng Support Vectors tăng gấp đôi (khiến thời gian inference tăng từ $1.93\text{ ms}$ lên $>3.5\text{ ms/ảnh}$), bạn sẽ chọn mô hình nào để triển khai thực tế trên vi xử lý nhúng (như Raspberry Pi hay NVIDIA Jetson)? Hãy đưa ra phán quyết kỹ sư của bạn!

---
**Chúc bạn thực hiện thực nghiệm tinh chỉnh sắc bén, bứt phá mọi giới hạn hiệu năng và hoàn thành xuất sắc mảnh ghép đỉnh cao cho Phase 2!** 🎯🔥
