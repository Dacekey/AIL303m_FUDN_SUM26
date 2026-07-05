# NHIỆM VỤ M04: Thực thi Thí nghiệm 3 - Đột phá Hoàn thiện Mô hình Final & Đánh giá Test (Final Polish & Evaluation)

- **Mã nhiệm vụ:** `M04`
- **Ngày ban hành:** 29/06/2026
- **Người thực hiện phụ trách:** Thành viên 4 (ML & Pipeline Engineer)
- **Giai đoạn thực hiện:** Giai đoạn 2 (Bấm chạy cuối cùng, chốt kết quả toàn dự án)
- **Đầu ra bắt buộc (Deliverables):** Mô hình Final hoàn thiện (có PCA + Augmentation) và Bảng kết quả đánh giá duy nhất 1 lần trên tập `Test Split`.

---

## 1. MỤC TIÊU NHIỆM VỤ

Bạn là người giữ trọng trách chốt hạ toàn bộ dự án. Nhiệm vụ của bạn là kiểm chứng **Giả thuyết H3** trong `LOG-03`: *Chứng minh rằng việc tích hợp giảm chiều PCA và kỹ thuật Tăng cường dữ liệu trên ảnh (`Minority Augmentation`) vừa giúp mô hình suy luận siêu nhanh, vừa cứu vãn điểm Recall cho các lớp biển báo hiếm gặp.* Sau đó tiến hành chạy kiểm thử lần cuối trên tập `test`.

---

## 2. HƯỚNG DẪN THỰC THI CHI TIẾT TỪNG BƯỚC

### Bước 1: Tiếp nhận Đầu vào từ M01 và M03
* Nhận **Danh sách các class thiểu số** từ Thành viên 1 (M01).
* Nhận **Bộ đặc trưng chiến thắng** từ Thành viên 3 (M03, ví dụ: `HOG Gray + Color Hist`).

### Bước 2: Thiết lập Pipeline Tăng cường Dữ liệu Thiểu số (`Minority Augmentation`)
> ⚠️ **CẢNH BÁO KHOA HỌC NGHIÊM NGẶT:** Tuyệt đối **KHÔNG sử dụng thuật ngữ hay kỹ thuật SMOTE** trong code và báo cáo. Lý do: SMOTE nội suy tuyến tính trong không gian vector HOG nghìn chiều sẽ tạo ra các mẫu rác làm phá vỡ cấu trúc hình học vật lý của biển báo (theo phân tích chốt tại `LOG-03`).

* **Cách làm chuẩn:** Bạn áp dụng kỹ thuật gia tăng mẫu trực tiếp trên không gian ảnh 2D (`minority_light`) trước khi rút đặc trưng:
  * Lấy lọc riêng các ảnh thuộc danh sách class thiểu số trong tập `train`.
  * Sử dụng thư viện `Albumentations` hoặc `cv2` để tạo thêm 2-3 biến thể cho mỗi ảnh: xoay nhẹ $\pm 10^\circ$, điều chỉnh độ sáng $\pm 15\%$, lật ngang (chỉ áp dụng với biển đối xứng).
  * Rút đặc trưng HOG+Hist cho các ảnh mới này và gộp vào tập `train`.

### Bước 3: Thiết lập Pipeline Giảm chiều PCA & Huấn luyện (2-3 Lần chạy)
* Xây dựng pipeline hoàn chỉnh:
  $$\mathbf{StandardScaler()} \longrightarrow \mathbf{PCA}(\text{n\_components}=0.95) \longrightarrow \mathbf{SVC}\big(\text{kernel='rbf'}, C=10.0, \gamma=\text{'scale'}, \text{class\_weight='balanced'}\big)$$
* Chạy đánh giá trên tập `val` để so sánh với kết quả gốc của TN2:

| Mã TN | Cấu hình Thí nghiệm | Số chiều sau PCA | Val Accuracy (%) | Val Macro F1 (%) | Thời gian Inference (ms/ảnh) | Nhận xét |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **TN3.1** | Gốc từ TN2 (Không PCA, Không Aug) | *ví dụ: 2,276* | *chép từ M03* | *chép từ M03* | *...* | *Mô hình baseline từ bước trước* |
| **TN3.2** | Thêm PCA 95% Variance | *ví dụ: ~350* | *...* | *...* | **[Siêu nhanh]** | *Nén >80% số chiều, giữ vững F1* |
| **TN3.3** | Thêm PCA + `Minority Augmentation` | *ví dụ: ~350* | *...* | **[Đỉnh cao nhất]** | *...* | *Cải thiện Recall cho lớp hiếm* |

### Bước 4: MỞ KHÓA TẬP TEST & ĐÁNH GIÁ CHÍNH THỨC 1 LẦN DUY NHẤT
* Lấy mô hình chiến thắng đỉnh cao nhất (TN3.3), tiến hành chạy `predict` trên tập **`test split` (851 ảnh)** chưa từng được chạm vào từ đầu dự án đến nay.
* Xuất ra bảng `classification_report` và vẽ `confusion_matrix` cuối cùng.

---

## 3. BÀN GIAO & KẾT NỐI (FINAL HANDOVER)

Chúc mừng bạn! Ngay khi có số liệu trên tập Test, bạn báo cáo cho **Trưởng nhóm (Thành viên 1)** và triệu tập cả nhóm để:
1. Đưa toàn bộ bảng số liệu từ TN1, TN2, TN3 vào luận văn/báo cáo khoa học.
2. Ăn mừng vì nhóm đã hoàn thành một bài nghiên cứu thị giác máy tính cực kỳ chuyên sâu, chặt chẽ và không thể phản bác! 🏆
