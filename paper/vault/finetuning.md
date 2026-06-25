### 1. Hướng tinh chỉnh chung cho mô hình phân loại (SVM Tuning)                      
                                                                                      
Bất kể bạn dùng feature nào trong 4 bộ trên, mô hình SVM vẫn có thể được tối ưu hóa   
qua GridSearch hoặc RandomSearch:                                                     
                                                                                      
• Tuning Hyperparameters: Thử nghiệm với các kernel khác nhau (Linear, RBF,           
Polynomial). Nếu dùng RBF, việc dò tìm cặp giá trị  C  (độ phạt lỗi) và  gamma  (tầm  
ảnh hưởng của 1 điểm dữ liệu) là bắt buộc để tìm ra điểm tối ưu nhất.                 
• Xử lý mất cân bằng dữ liệu (Class Imbalance): Ở các bước trước ta thấy dataset có   
class > 1000 ảnh, nhưng có class chỉ ~20 ảnh. Dù đã dùng tính năng tự động cân bằng   
(vd:  class_weight='balanced' ), ta có thể can thiệp sâu hơn bằng kỹ thuật Over-      
sampling (SMOTE) kết hợp với các feature này trước khi đưa vào SVM.                   
                                                                                      
### 2. Tinh chỉnh riêng cho từng bộ Feature                                           
                                                                                      
• Raw Pixels (Baseline):                                                              
    • Kích thước ảnh: Hiện tại ảnh đang được resize về một mức cố định (ví dụ 64x64)  
    rồi dàn phẳng (flatten). Ta có thể tuning thử các size khác (32x32, 48x48,        
    128x128). Size quá to gây nhiễu và chậm, size quá nhỏ gây mất chi tiết.           
    • Giảm chiều dữ liệu (PCA): Dàn phẳng pixel tạo ra vector có số chiều khổng lồ (ví
    dụ ảnh 64x64 sinh ra 4096 features). Áp dụng PCA để giữ lại 95% phương sai        
    (variance) sẽ giúp SVM chạy nhanh hơn, loại bỏ nhiễu nền và chống over-fitting cực
    kỳ hiệu quả.                                                                      
    • Scaling: Thử nghiệm giữa  StandardScaler  và  MinMaxScaler  trên các giá trị    
    pixel.                                                                            
• HOG only (yuv) & HOG + Color Histogram (gray / yuv):                                
    • Tuning tham số HOG: Đặc trưng HOG rất nhạy cảm với các thông số truyền vào. Có  
    thể tuning các giá trị:  orientations  (số hướng gradient, thử 8, 9, 12),         
    pixels_per_cell  (thử 4x4, 8x8) và  cells_per_block  (thử 1x1, 2x2).              
    • Trọng số cho YUV: Khi trích xuất HOG trên không gian YUV, kênh Y (Luma/độ sáng) 
    chứa nhiều thông tin về hình khối vật thể nhất, trong khi U và V thiên về màu sắc.
    Có thể tính HOG trên kênh Y chi tiết hơn (hoặc gán trọng số cao hơn) thay vì gộp  
    chung cả 3 kênh một cách cào bằng.                                                
    • Tuning Color Histogram: Thử nghiệm số lượng  bins  (giỏ) khác nhau (ví dụ: 8, 16,
    32) cho mỗi kênh HSV.                                                             
• Vấn đề ghép nối (Concatenation) HOG + Histogram:                                    
    • Giá trị sinh ra từ HOG và giá trị của Histogram khác hẳn nhau về dải số học. Nếu
    chỉ nối (concatenate) 2 vector này lại, SVM có thể sẽ "thiên vị" đặc trưng nào có 
    giá trị lớn hơn. Bắt buộc phải có bước Normalization/Scaling độc lập cho từng bộ  
    feature rồi mới ghép nối.                                                         
                                                                                      
                                                                                      
### 3. Hướng phát triển dữ liệu (Data-level Development)                              
                                                                                      
• Tiền xử lý ảnh (Image Preprocessing): Do ảnh được crop bằng bounding box, tỷ lệ     
khung hình (aspect ratio) của vật thể gốc có thể bị biến dạng khi ép về hình vuông.   
Hướng phát triển tốt là Padding (thêm viền đen) để ảnh thành hình vuông trước khi     
resize thay vì bóp méo ảnh.                                                           
• Data Augmentation: Tăng cường dữ liệu cho các class thiểu số (thiếu ảnh) bằng cách
lật ngang, xoay nhẹ, đổi độ sáng, thêm nhiễu Gaussian... trên tập Train.              

### 4. Hướng phát triển nâng cao (Deep Learning)

Nếu giới hạn của việc trích xuất đặc trưng thủ công (Hand-crafted features như
HOG/Hist) đã tới trần:                                                                

• Bỏ qua SVM và dùng luôn CNN (Mạng nơ-ron tích chập). Những mô hình siêu nhẹ như     
MobileNet  hay  ResNet18  tự động học tính năng (feature learning) từ Raw Pixels hiệu 
quả hơn con người tự định nghĩa HOG hay Histogram rất nhiều, và cực kì phù hợp với bài
toán Image Classification này.
