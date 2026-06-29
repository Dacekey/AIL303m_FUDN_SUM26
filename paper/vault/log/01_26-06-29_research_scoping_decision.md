# LOG-01: Quyết định Thu hẹp Phạm vi & Tái cấu trúc Nghiên cứu (Research Scoping Decision)

- **Ngày ghi nhận:** 29/06/2026
- **Người thực hiện:** Nhóm nghiên cứu & AI Assistant
- **Trạng thái:** Đã phê duyệt (Approved & Locked)
- **Danh mục:** Phương pháp luận nghiên cứu (Research Methodology)

---

## 1. BỐI CẢNH & VẤN ĐỀ ĐẶT RA (CONTEXT & PROBLEM STATEMENT)

Sau khi hoàn thành thực nghiệm ở **Phase 1** (Khảo sát 10 biến thể đặc trưng truyền thống) và **Phase 2** (Khảo sát bổ sung 4 phương pháp Edge Histogram, Gabor, LBP, Hu Moments), nhóm nghiên cứu đứng trước hai thách thức phương pháp luận lớn:

1. **Sự hoài nghi về tính công bằng của Baseline ban đầu (Methodological Bias Critique):**
   Toàn bộ bảng xếp hạng hiệu năng ở Phase 1 & 2 được đánh giá trên một bộ cấu hình baseline cố định (Kích thước `64x64`, resize bóp méo `stretch`, HOG cell `8x8`, bộ phân loại `StandardScaler + SVM RBF C=10`). Dù cấu hình này có cơ sở lý luận vững chắc từ tài liệu kinh điển (*Dalal & Triggs, 2005*) và phù hợp với tiêu chuẩn biển báo Việt Nam (*QCVN 41:2019/BGTVT*), nhưng việc áp đặt nó làm thước đo chung có nguy cơ thiên vị HOG và vô tình làm suy giảm tiềm năng của các phương pháp khác (như LBP hay Gabor).
2. **Nguy cơ bùng nổ quy mô tổ hợp (Combinatorial Explosion):**
   Nếu tiến hành Fine-tune toàn bộ 14 phương pháp cùng với các biến thể tiền xử lý ảnh, giảm chiều và tham số học máy, không gian tìm kiếm sẽ lên tới **36,068 cấu hình** (hoặc ~72,000 nếu bật SMOTE). Việc gộp chung các trục vấn đề vào một vòng lặp Random Search (120 mẫu) tuy tiết kiệm thời gian chạy code nhưng làm bài nghiên cứu bị dàn trải, loãng trọng tâm ("mile wide, inch deep") và cực kỳ khó giải thích mối quan hệ nhân quả trong luận văn/báo cáo khoa học.

---

## 2. CÁC QUYẾT ĐỊNH CHIẾN LƯỢC (STRATEGIC DECISIONS)

Để đảm bảo chiều sâu khoa học, tính mạch lạc trong logic phản biện và tối ưu nguồn lực tính toán, nhóm nghiên cứu thống nhất thực hiện **3 Quyết định Tái cấu trúc**:

### Quyết định 1: Cắt bỏ (Prune) hoàn toàn Phase 2 khỏi giai đoạn Fine-tuning
* **Hành động:** Loại bỏ Gabor, LBP, Edge Histogram và Hu Moments khỏi Phase 3. Trong bài báo cáo chính thức, kết quả Phase 2 chỉ được trình bày gọn trong 1 bảng số liệu ở phần "Khảo sát ban đầu".
* **Lý luận bảo vệ:** Thực nghiệm chứng minh các đặc trưng kết cấu vi mô (LBP - F1 ~30%), mô-men toàn cục (Hu Moments - F1 ~25%) và bộ lọc tần số thô (Gabor/Edge Hist - F1 ~58%) hoàn toàn thất bại trước biểu diễn hình học phức tạp của biển báo Việt Nam so với nhóm HOG (>92%). Việc cắt bỏ giúp tập trung 100% nguồn lực vào các đặc trưng tiềm năng nhất.

### Quyết định 2: Khóa cố định (Lock) Mô hình Học máy
* **Hành động:** Khóa cố định bộ phân loại ở chuẩn mực: **`StandardScaler` + `SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced')`**. Không thực hiện Grid Search tham số SVM phức tạp trong Phase 3.
* **Lý luận bảo vệ:** Việc khóa bộ phân loại giúp tạo ra một hệ quy chiếu ổn định, loại bỏ sự nhiễu loạn tham số mô hình để toàn bộ bài nghiên cứu tập trung giải quyết sâu sắc 2 câu hỏi cốt lõi về bản chất Dữ liệu và Đặc trưng biển báo giao thông Việt Nam.

### Quyết định 3: Chuyển đổi từ Random Search gộp chung sang 3 Thí nghiệm Chuyên đề Tuần tự
Thay vì chạy 120 mẫu ngẫu nhiên gộp chung khó diễn giải, Phase 3 được chia thành lộ trình **3 Thí nghiệm rành mạch (Thematic Stepwise Experiments)**, bước sau kế thừa kết quả tốt nhất của bước trước:

* **Thí nghiệm 1 (RQ1 - Tối ưu Tiền xử lý & Kích thước):**
  * *Mục tiêu:* Xác định cách resize và độ phân giải tối ưu cho biển báo VN.
  * *Setup:* Lấy `HOG Gray` chạy trên ma trận 3 Kích thước (`32x32`, `48x48`, `64x64`) $\times$ 2 Chế độ resize (`stretch` ép vuông vs `pad_square` đệm viền đen giữ tỷ lệ hình học). *(6 lần chạy)*
* **Thí nghiệm 2 (RQ2 - Tối ưu Kết hợp Đặc trưng & Màu sắc):**
  * *Mục tiêu:* Tìm sự kết hợp tối ưu giữa đường nét hình học và thông tin màu sắc mà không bị Lời nguyền số chiều.
  * *Setup:* Trên tiền xử lý tốt nhất chốt từ TN1, so sánh 3 bộ đặc trưng: `HOG Gray`, `HOG YUV`, `HOG Gray + Color Hist (HSV)`. *(3 lần chạy)*
* **Thí nghiệm 3 (Đột phá Hoàn thiện Mô hình Final):**
  * *Mục tiêu:* Tăng tốc độ suy luận (inference) và cải thiện độ chính xác cho các lớp thiểu số.
  * *Setup:* Lấy bộ đặc trưng chiến thắng từ TN2, áp dụng **PCA giảm chiều** (giữ 95% variance) và kỹ thuật **Data Augmentation nhẹ cho class thiểu số (`minority_light`)**. *(2-3 lần chạy)*

---

## 3. TÁC ĐỘNG & KẾT QUẢ KỲ VỌNG (EXPECTED IMPACT)

1. **Về mặt tính toán:** Giảm số lượng cấu hình cần huấn luyện từ hàng ngàn (hoặc 120 cấu hình random) xuống chỉ còn khoảng **12 - 15 cấu hình** tuần tự rõ ràng. Thời gian chạy thực nghiệm giảm xuống dưới 10 phút.
2. **Về mặt chất lượng bài báo/luận văn:** 
   * Cấu trúc bài viết mạch lạc, kể một câu chuyện nghiên cứu khoa học có chiều sâu.
   * Mỗi bảng số liệu minh chứng cho một kết luận cụ thể, triệt tiêu hoàn toàn các điểm mù phản biện của hội đồng về việc gộp chung tham số hay thiên vị cấu hình.
