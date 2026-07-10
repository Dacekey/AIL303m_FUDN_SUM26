# BÁO CÁO M05: Thực thi Khảo sát & Đối chứng Các Mô hình Học máy Phân loại (Classical ML Models Benchmarking & Ablation Study)
* Ngày ban hành: 03/07/2026
* Giai đoạn thực hiện: Giai đoạn Mở rộng / Thực nghiệm Đối chứng (Phase 2)

## Bản chất Hình học của Không gian Đặc trưng Sau PCA ($D = 571$)

Trước khi đi vào đánh giá chi tiết hiệu năng của từng mô hình, việc hiểu rõ bản chất không gian đầu vào là yếu tố tiên quyết. Tập dữ liệu của chúng ta là sự kết hợp của các đặc trưng toàn cục và cục bộ (HOG, Grayscale, Color Histogram). Sau khi áp dụng thuật toán Phân tích Thành phần Chính (PCA) để giữ lại 95% phương sai, dữ liệu đã được nén xuống không gian $571$ chiều ($\mathbb{R}^{571}$).

* **Phép xoay trục hệ tọa độ (Orthogonal Rotation):** PCA không chỉ đơn thuần là giảm chiều. Ma trận hiệp phương sai của tập huấn luyện $S$ được phân tích để tìm ra 571 eigenvector tương ứng với các eigenvalue lớn nhất. Hệ quả hình học là các chiều không gian mới không còn mang ý nghĩa độc lập (như một bin màu hay một hướng cạnh cụ thể) mà là một tổ hợp tuyến tính của toàn bộ đặc trưng gốc. Việc không gian bị "xoay" làm thay đổi hoàn toàn hình dáng của ranh giới phân tách (decision boundary) ban đầu.
* **Mật độ phân tán:** Dù đã giảm chiều, $\mathbb{R}^{571}$ vẫn là một không gian có số chiều cực kỳ cao. Tại đây, dữ liệu bị phân tán thưa thớt, tạo ra những thách thức toán học nghiêm trọng cho một số thuật toán nhất định.

---

### 1. Vì sao SVM RBF vẫn giữ vị thế thống trị (Hoặc nhỉnh hơn rõ rệt)?
Dựa trên biểu đồ "Comparison of Tuned Macro F1 Scores" và bảng Benchmark, mô hình Support Vector Classifier với kernel RBF (Radial Basis Function) vượt trội hơn hẳn phần còn lại. Sự thống trị này bắt nguồn từ bản chất tối ưu lề và khả năng ánh xạ phi tuyến.

   * Cơ chế Support Vectors (Tối ưu hóa lề):
        Thuật toán SVM không cố gắng mô hình hóa toàn bộ phân phối của dữ liệu, mà chỉ tập trung tìm một siêu phẳng (hyperplane) phân cách có lề lớn nhất (Maximum Margin). Hàm mục tiêu của SVM được định nghĩa như sau:
        $$\min_{w, b, \xi} \frac{1}{2}||w||^2 + C \sum_{i=1}^{n} \xi_i$$
        Trong không gian liên tục 571 chiều của gradient HOG, dữ liệu có thể rất nhiễu. Nhờ việc chỉ dựa vào các điểm ranh giới (Support Vectors), SVM bỏ qua các điểm nằm sâu bên trong các cụm phân lớp, giúp mô hình miễn nhiễm với phần lớn nhiễu (noise) nội bộ.

   * Thủ thuật Kernel RBF (Ánh xạ vô hạn chiều):
        Biển báo giao thông trong thực tế chịu biến dạng hình học phi tuyến rất lớn (do góc chụp 3D, ánh sáng, vật cản). Một siêu phẳng tuyến tính thông thường không thể cắt chia rạch ròi. Kernel RBF giải quyết vấn đề này bằng cách tính tích vô hướng ngầm định thông qua hàm Gauss:
        $$K(x_i, x_j) = \exp\left(-\frac{||x_i - x_j||^2}{2\sigma^2}\right) = \exp(-\gamma ||x_i - x_j||^2)$$
        Về mặt hình học, Kernel RBF sử dụng chuỗi Taylor để chiếu dữ liệu từ không gian 571 chiều lên một không gian Hilbert có số chiều vô hạn (Infinite-dimensional Hilbert Space). Trong không gian này, dữ liệu trở nên phân tách tuyến tính. Khi chiếu ngược về không gian gốc, ranh giới quyết định (decision boundary) trở thành các mặt cong phức tạp, có khả năng "ôm sát" và bao bọc các biến dạng hình học lõm/lồi của các cụm biển báo mà không gây ra hiện tượng Overfitting.

---

### 2. Vì sao nhóm Cây quyết định (Random Forest / LightGBM) gặp hạn chế?
Mặc dù là các thuật toán Ensemble rất mạnh, Random Forest và LightGBM lại ghi nhận mức Macro F1 thấp nhất (~0.84 - 0.86). Vấn đề nằm ở sự bất đồng nhất giữa hình học của cây và bản chất đặc trưng.

   * Mặt phẳng phân chia trực giao (Axis-aligned orthogonal splits):
        Về mặt toán học, tại mỗi node, thuật toán cây ra quyết định bằng cách chia không gian theo một ngưỡng vô hướng trên một trục duy nhất. Biên quyết định có dạng:
        $$x_d \le \theta$$
        Hình học hóa, điều này nghĩa là cây chỉ có thể dùng các nhát cắt hoàn toàn vuông góc với các trục tọa độ. 

   * Hiện tượng Phân mảnh vùng quyết định (Decision Boundary Fragmentation):
        Các vector HOG là các tổ hợp tuyến tính của gradient, tạo ra các ranh giới phân bố theo các góc chéo trong không gian. Để cắt một "đường chéo phi tuyến" bằng các lát cắt vuông góc, cây phải phân nhánh thành vô số bậc thang nhỏ vụn $\rightarrow$ Gây ra hiện tượng phân mảnh vùng quyết định (Decision Boundary Fragmentation), dẫn đến Overfit trên tập Train và suy luận chậm do phải duyệt qua hàng trăm tầng cây. Sự phân mảnh này cũng làm suy giảm độ tin cậy của xác suất dự đoán và làm giảm khả năng khái quát hóa hình học liên tục của đặc trưng ảnh.

---

### 6.3. Vì sao K-Nearest Neighbors (KNN) sa sút trong không gian này?
Mặc dù KNN có F1 Test khá ổn định trong bài toán này (0.9423), nhưng về mặt bản chất lý thuyết, thuật toán này luôn bị đe dọa bởi tính chất đa chiều của dữ liệu HOG.

   * Lời nguyền cô đặc khoảng cách (Distance Concentration Curse):
        Khoảng cách Euclid cơ bản giữa hai vector $x, y \in \mathbb{R}^D$ được tính bằng:
        $$d(x, y) = \sqrt{\sum_{i=1}^{D} (x_i - y_i)^2}$$
        Tuy nhiên, theo định lý giới hạn trong không gian đa chiều (Dimensionality Asymptotics), khi số chiều $D$ lớn (ở đây $D=571$), phương sai của sự chênh lệch khoảng cách tiến tới 0. Định lý chỉ ra rằng tỉ số giữa khoảng cách xa nhất ($D_{max}$) và gần nhất ($D_{min}$) từ một điểm truy vấn đến các điểm dữ liệu sẽ tiến về 0:
        $$\lim_{D \to \infty} \frac{D_{max} - D_{min}}{D_{min}} = 0$$
        Nói cách hình học, khi bạn đứng ở bất kỳ đâu trong không gian 571 chiều, khoảng cách từ bạn đến *tất cả mọi điểm* xung quanh dường như là bằng nhau. Các điểm nằm ở "rìa" hay "trung tâm" của cụm đều trông xấp xỉ khoảng cách với nhau. Điều này khiến cơ chế "bầu chọn lân cận gần nhất" của KNN bị mất phương hướng khi bầu chọn lân cận.

---

### 6.4. Vì sao Logistic Regression bị giới hạn siêu phẳng?
Logistic Regression cho kết quả rất đáng nể (F1: 0.9482) với tốc độ suy luận nhanh nhất (0.0019s). Tuy nhiên, nó vấp phải trần giới hạn về mặt toán học so với các mô hình phi tuyến.

   * Giới hạn của Hàm phân loại tuyến tính toàn cục:
        Phương trình phân loại của Multinomial Logistic Regression dựa trên hàm Softmax, với ranh giới quyết định giữa hai lớp $k$ và $j$ là nơi có tỷ lệ log-odds bằng 0, tương đương với phương trình siêu phẳng:
        $$(\mathbf{w}_k - \mathbf{w}_j)^T \mathbf{x} + (b_k - b_j) = 0$$
        Đây là một phương trình tuyến tính tuyệt đối ($w^Tx + b = 0$). Về mặt hình học, nó tương đương với việc dùng một "tấm ván phẳng" khổng lồ để cắt không gian 571 chiều ra làm các nửa không gian (half-spaces).
        
   * Sự cứng nhắc trước cụm lõm/phức tạp:
        Ranh giới siêu phẳng cứng nhắc không thể phân tách các cụm biển báo bị xô lệch góc chụp 3D ngoài trời (vốn có hình dạng cụm lồi lõm phức tạp). Nếu cố gắng chia cắt điểm này, siêu phẳng bắt buộc phải cắt phạm vào phân phối của điểm khác ở đầu kia của không gian, dẫn đến một lượng sai số phân loại không thể tránh khỏi. Nó hoàn toàn không có khả năng "uốn lượn" cục bộ như SVC(RBF).
