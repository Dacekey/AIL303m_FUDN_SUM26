# NHIỆM VỤ M03: Thực thi Thí nghiệm 2 - Khảo sát Tối ưu Không gian màu & Dung hợp Đặc trưng (Colorspace & Fusion Experiment)

- **Mã nhiệm vụ:** `M03`
- **Ngày ban hành:** 29/06/2026
- **Người thực hiện phụ trách:** Thành viên 3 (Feature Extraction Engineer)
- **Giai đoạn thực hiện:** Giai đoạn 2 (Bấm chạy thứ hai, ngay sau khi nhận cấu hình chốt từ Thành viên 2)
- **Đầu ra bắt buộc (Deliverables):** Bảng số liệu chứng minh nghịch lý nhiễu YUV và Chốt bộ đặc trưng chiến thắng bàn giao cho Thành viên 4.

---

## 1. MỤC TIÊU NHIỆM VỤ

Bạn phụ trách kiểm chứng **Giả thuyết H2** trong `LOG-03`: *Chứng minh rằng sự kết hợp giữa đường nét HOG trên kênh xám và màu sắc HSV (`HOG Gray + Color Hist`) là tối ưu nhất, đánh bại việc trích xuất HOG trên 3 kênh màu (`HOG YUV`) do tránh được nhiễu sắc độ U/V và Lời nguyền số chiều.*

---

## 2. HƯỚNG DẪN THỰC THI CHI TIẾT TỪNG BƯỚC

### Bước 1: Tiếp nhận Đầu vào từ Nhiệm vụ M02
* Bạn phải chờ Thành viên 2 xác nhận cấu hình Tiền xử lý chiến thắng (Ví dụ: `64x64 + pad_square`).
* **Tuyệt đối không tự ý đổi size ảnh hay cách resize** để đảm bảo tính nhất quán kế thừa khoa học!

### Bước 2: Thiết lập Ma trận Thí nghiệm (3 Lần chạy)
* Sử dụng ảnh đã qua tiền xử lý chiến thắng từ M02, bạn viết hàm rút đặc trưng cho 3 ứng cử viên mạnh nhất từ trước đến nay:
  1. **`HOG Gray`:** Chuyển ảnh sang Grayscale, trích xuất HOG chuẩn.
  2. **`HOG YUV`:** Chuyển ảnh sang hệ màu YUV, rút HOG riêng trên 3 kênh Y, U, V rồi nối tiếp lại (`np.concatenate`).
  3. **`HOG Gray + Color Hist (HSV)`:** Rút `HOG Gray`, sau đó rút thêm Color Histogram 3D (`bins=(8,8,8)`) trên không gian HSV đã chuẩn xác suất (`hist /= hist.sum() + 1e-8`), nối ghép mảng (`np.hstack`).
* Khóa cố định bộ phân loại là **`StandardScaler() + SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced')`** (theo đúng `LOG-02`).
* Chạy huấn luyện trên tập `train` và đánh giá trên tập `val`:

| Mã TN | Bộ Đặc trưng | Số chiều ($n\_features$) | Val Accuracy (%) | Val Macro F1 (%) | Thời gian Train (s) | Nhận xét Khoa học |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **TN2.1** | `HOG Gray` | *ví dụ: 1,764* | *...* | *...* | *...* | *Nhanh, gọn nhưng thiếu màu* |
| **TN2.2** | `HOG YUV` | *ví dụ: 5,292* | *...* | *...* | *...* | *Nặng gấp 3 lần, kiểm chứng có bị nhiễu U/V tụt F1 không* |
| **TN2.3** | `HOG Gray + Color Hist` | *ví dụ: 2,276* | *...* | **[Kỳ vọng cao nhất]** | *...* | *Cân bằng hoàn hảo hình học & màu sắc* |

### Bước 3: Phân tích & Viết Kết luận
* Trình bày rõ ràng trong báo cáo: Việc tăng số chiều lên $>5000$ của `HOG YUV` có mang lại lợi ích gì không hay chỉ làm chậm mô hình?
* Khẳng định sự vượt trội của dung hợp đa thức (`HOG Gray + Hist`).

---

## 3. BÀN GIAO & KẾT NỐI (HANDOVER)

Ngay khi hoàn thành bảng số liệu trên, bạn bàn giao lập tức cho **Thành viên 4**:
* Chốt bằng văn bản: **"Bộ đặc trưng chiến thắng chính thức của dự án là: [HOG Gray + Color Hist HSV hay HOG YUV] với số chiều gốc là [????]"**.
* Chuyển mảng ma trận đặc trưng `X_train`, `X_val` (hoặc hàm trích xuất chuẩn xác) sang cho Thành viên 4 để bước vào khâu đột phá cuối cùng!

**Chúc bạn kiểm chứng thành công và chốt hạ bộ đặc trưng vô địch cho nhóm!** 🚀
