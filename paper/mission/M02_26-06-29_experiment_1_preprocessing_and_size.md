# NHIỆM VỤ M02: Thực thi Thí nghiệm 1 - Khảo sát Tối ưu Tiền xử lý & Kích thước ảnh (Preprocessing & Resolution Experiment)

- **Mã nhiệm vụ:** `M02`
- **Ngày ban hành:** 29/06/2026
- **Người thực hiện phụ trách:** Thành viên 2 (Data Preprocessing Engineer)
- **Giai đoạn thực hiện:** Giai đoạn 2 (Bấm chạy đầu tiên ngay sau khi chuẩn bị xong code module)
- **Đầu ra bắt buộc (Deliverables):** Bảng số liệu so sánh 6 cấu hình Tiền xử lý và Chốt bộ tham số Kích thước/Resize chiến thắng bàn giao cho Thành viên 3.

---

## 1. MỤC TIÊU NHIỆM VỤ

Bạn phụ trách kiểm chứng **Giả thuyết H1** trong `LOG-03`: *Chứng minh rằng kỹ thuật đệm viền đen giữ nguyên tỷ lệ hình học (`pad_square`) vượt trội hơn việc ép vuông bóp méo (`stretch`), và xác định độ phân giải vàng (`32`, `48` hay `64`) cho biển báo Việt Nam.*

---

## 2. HƯỚNG DẪN THỰC THI CHI TIẾT TỪNG BƯỚC

### Bước 1: Chuẩn bị Hàm Tiền xử lý `pad_square`
* Viết một hàm Python sử dụng OpenCV (`cv2`) nhận đầu vào là ảnh crop từ `cropped_dataset`:
  * **Chế độ `stretch` (Cũ):** Dùng `cv2.resize(img, (target_size, target_size))` trực tiếp.
  * **Chế độ `pad_square` (Mới):** Tính toán cạnh dài nhất $L = \max(W, H)$. Thu phóng ảnh sao cho cạnh dài nhất bằng `target_size`, cạnh ngắn hơn được đệm viền đen đối xứng bằng `cv2.copyMakeBorder` với giá trị pixel bằng 0 để ảnh trở thành hình vuông `target_size x target_size` mà **không bị bóp méo tỷ lệ**.

### Bước 2: Thiết lập Ma trận Thí nghiệm (6 Lần chạy)
* Khóa cố định bộ đặc trưng là **`HOG Gray`** (tham số chuẩn: `orientations=9`, `pixels_per_cell=(8,8)`, `cells_per_block=(2,2)`). *(Lưu ý: với size `32x32` có thể chỉnh `pixels_per_cell=(4,4)` nếu cần thiết để đủ số block).*
* Khóa cố định bộ phân loại là **`StandardScaler() + SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced')`** (theo đúng `LOG-02`).
* Chạy huấn luyện trên tập `train` và đánh giá trên tập `val` cho 6 cấu hình sau:

| Mã TN | Kích thước (`Size`) | Chế độ Resize | Số chiều HOG | Val Accuracy (%) | Val Macro F1 (%) | Thời gian Train (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TN1.1** | `32x32` | `stretch` | *tự động tính* | *...* | *...* | *...* |
| **TN1.2** | `32x32` | `pad_square` | *tự động tính* | *...* | *...* | *...* |
| **TN1.3** | `48x48` | `stretch` | *tự động tính* | *...* | *...* | *...* |
| **TN1.4** | `48x48` | `pad_square` | *tự động tính* | *...* | *...* | *...* |
| **TN1.5** | `64x64` | `stretch` | 1,764 | *...* | *...* | *...* |
| **TN1.6** | `64x64` | `pad_square` | 1,764 | *...* | **[Kỳ vọng cao nhất]** | *...* |

### Bước 3: Phân tích & Viết Kết luận
* So sánh từng cặp (ví dụ TN1.5 vs TN1.6) để chứng minh việc giữ tỷ lệ hình học (`pad_square`) tăng điểm Macro F1 bao nhiêu phần trăm.
* Kiểm tra xem kích thước `48x48` có giữ được điểm F1 ngang ngửa `64x64` nhưng train nhanh gấp đôi không.

---

## 3. BÀN GIAO & KẾT NỐI (HANDOVER)

Ngay khi chạy xong 6 cấu hình trên, bạn thực hiện bàn giao lập tức cho **Thành viên 3**:
* Chốt bằng văn bản: **"Cấu hình Tiền xử lý chiến thắng là: Size = [??x??], Mode = [pad_square hay stretch]"**.
* Gửi toàn bộ pipeline tiền xử lý hoặc tập dữ liệu ảnh đã resize theo cấu hình chiến thắng đó sang cho Thành viên 3 để chạy tiếp Thí nghiệm 2!

---
Checklist
- Audit Log
- Pipeline (Steps)
- Phan tich -> Ket luan
- Thu nghiem va bang chung chung minh

**Chúc bạn chạy code mượt mà và tìm ra cấu hình tiền xử lý vàng cho nhóm!** ⚡
