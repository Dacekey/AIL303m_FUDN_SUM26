# NHIỆM VỤ M01: Kiểm chứng Phân tích trước Tinh chỉnh & Chẩn đoán Lớp Thiểu số (Error Analysis & Minority Diagnosis)

- **Mã nhiệm vụ:** `M01`
- **Ngày ban hành:** 29/06/2026
- **Người thực hiện phụ trách:** Thành viên 1 (Trưởng nhóm / Data Analyst)
- **Giai đoạn thực hiện:** Giai đoạn 1 (Làm ngay từ đầu, song song với quá trình chuẩn bị code của TV2, TV3, TV4)
- **Đầu ra bắt buộc (Deliverables):** Danh sách mã biển báo bị lỗi nhiều nhất và danh sách class thiểu số cần Augmentation (bàn giao cho Thành viên 4).

---

## 1. MỤC TIÊU NHIỆM VỤ

Nhiệm vụ này đóng vai trò là "mắt thần" của cả dự án. Trước khi nhóm chạy các thí nghiệm tối ưu ở Phase 3, bạn cần mở các kết quả thực nghiệm cũ từ Phase 1 & 2 để **kiểm chứng bằng số liệu thực tế** các chẩn đoán trong `LOG-03`:
1. Có đúng là các lớp thiểu số (<30 ảnh) đang có Recall rất thấp không?
2. Có đúng là các biển báo hình tam giác/chữ nhật đang bị nhận diện nhầm do bóp méo hình học không?

---

## 2. HƯỚNG DẪN THỰC THI CHI TIẾT TỪNG BƯỚC

### Bước 1: Thu thập số liệu từ Phase 1 & 2
* Truy cập vào thư mục chứa kết quả cũ (ví dụ: `result/` hoặc `log/exphase_2_result/` hoặc các bảng classification report đã lưu từ `exphase_1.ipynb`).
* Tìm báo cáo phân loại (`val_classification_report`) của mô hình tốt nhất Phase 1: **`HOG + Color Histogram (gray)`** hoặc **`Raw Pixels`**.

### Bước 2: Sàng lọc Danh sách Lớp Thiểu số bị lỗi (Bottleneck Class List)
* Lọc ra tất cả các lớp biển báo (class ID, ví dụ `W.205a`, `P.103a`, `I.409`...) thỏa mãn một trong hai điều kiện sau:
  * **Điều kiện A (F1 / Recall thấp):** Điểm `Recall` hoặc `F1-score` trên tập Validation $< 60\%`.
  * **Điều kiện B (Mất cân bằng nghiêm trọng):** Số lượng mẫu hỗ trợ (`support`) trong tập Train/Val cực ít (ví dụ $< 30$ ảnh).
* Lập bảng thống kê theo mẫu sau vào một file báo cáo nhanh:

| Mã lớp (Class ID) | Tên biển báo | Số ảnh Train | Recall Val (%) | F1 Val (%) | Chẩn đoán nguyên nhân lỗi |
| :---: | :--- | :---: | :---: | :---: | :--- |
| *Ví dụ: W.205a* | *Đường giao nhau* | *18* | *35.0%* | *42.1%* | *Thiếu mẫu trầm trọng (Long-tailed)* |
| *Ví dụ: I.409* | *Chỉ dẫn xe buýt* | *45* | *52.0%* | *58.0%* | *Biển chữ nhật dài bị bóp méo khi ép vuông 64x64* |

### Bước 3: Phân tích Top nhầm lẫn (Top Confusions Analysis)
* Mở ma trận nhầm lẫn (`val_confusion_matrix` hoặc `val_top_confusions`).
* Trả lời câu hỏi: *Các biển báo bị nhầm với nhau thuộc nhóm nào?*
  * Nếu nhầm giữa `40` và `50` $\rightarrow$ Lỗi do độ phân giải ảnh chưa đủ nét để đọc chữ số.
  * Nếu nhầm giữa biển tam giác cảnh báo và biển tròn cấm $\rightarrow$ Lỗi do mất tỷ lệ hình học (`stretch`).

---

## 3. BÀN GIAO & KẾT NỐI (HANDOVER)

Ngay sau khi hoàn thành bảng thống kê trên (dự kiến trong vòng 2-4 giờ làm việc), bạn thực hiện 2 việc:
1. Gửi **Danh sách các mã lớp bị bóp méo hình học** cho **Thành viên 2** để củng cố giả thuyết cho Thí nghiệm 1 (`pad_square`).
2. Gửi **Danh sách các mã lớp thiểu số (Recall < 60% hoặc Support < 30)** cho **Thành viên 4** để thành viên này đưa vào script cấu hình tăng cường dữ liệu (`Minority Augmentation`) cho Thí nghiệm 3.

**Chúc bạn hoàn thành nhiệm vụ xuất sắc và mở đường chuẩn xác cho cả nhóm!** 🎯
