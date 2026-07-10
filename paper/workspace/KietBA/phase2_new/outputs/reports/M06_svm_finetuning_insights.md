# M06 SVM Fine-tuning Insights

## 1. Bản chất hình học của C và gamma trong không gian PCA

Notebook sử dụng feature cache từ M04 với chế độ: `reconstructed_scaler_pca_from_m04_feature_cache`.

Không gian train sau PCA có shape: `(7035, 571)`.

PCA variance: `0.9500`.

Best Tuned SVM Final dùng:

- Kernel: `linear`
- Degree: `nan`
- C: `0.1`
- Gamma: `N/A`
- Class weight: `None`

Trong SVM, `C` điều khiển mức phạt khi phân loại sai. `C` lớn làm margin cứng hơn, model cố gắng sửa nhiều điểm sai hơn. `C` nhỏ cho phép margin mềm hơn, giảm rủi ro overfitting.

Với RBF, Polynomial và Sigmoid kernel, `gamma` điều khiển độ rộng vùng ảnh hưởng của từng support vector. `gamma` lớn tạo biên quyết định cục bộ, dễ bắt chi tiết nhưng dễ overfit. `gamma` nhỏ làm biên mượt hơn, phù hợp khi PCA đã giảm nhiễu.

## 2. RBF vs Polynomial trên HOG + Color Hist HSV

Kernel exploration và grid search được chọn bằng Validation Macro F1.

RBF thường phù hợp với traffic sign vì nó mô hình hóa các vùng ảnh hưởng cục bộ quanh support vectors. Điều này giúp xử lý tốt các biến thiên như ánh sáng, phối cảnh, độ méo và khác biệt nhỏ giữa các biển báo.

Polynomial degree 2 hoặc degree 3 có thể nắm bắt tương tác giữa hướng cạnh HOG và màu HSV. Tuy nhiên, nếu các tương tác này không ổn định giữa các ảnh, Polynomial kernel dễ tạo biên quyết định phức tạp và kém tổng quát hơn.

Kết quả final test:

- Test Accuracy: `96.12%`
- Test Macro F1: `96.29%`
- Support vectors: `4514`
- Inference speed: `1.5194648649701115 ms/image`

## 3. Bottleneck classes và nhầm lẫn tốc độ

Notebook thử `custom_dict` cho các lớp bottleneck nếu các lớp đó có trong dataset.

Bottleneck labels tìm thấy trong dataset:

W.205c, P.127_60, P.127_80, W.224, W.245a

Top confusion trên final test:

- `P.127_60` → `P.127_80`: 5 samples
- `P.130` → `P.131a`: 4 samples
- `P.131a` → `P.130`: 4 samples
- `P.127_80` → `P.127_60`: 2 samples
- `W.207` → `W.224`: 2 samples
- `W.224` → `W.245a`: 2 samples
- `P.103a` → `P.123b`: 1 samples
- `P.106a_Xe tải` → `P.107`: 1 samples
- `P.111` → `B.8a`: 1 samples
- `P.123b` → `P.127_40`: 1 samples

Nếu các lớp như `W.205c`, `P.127_60`, `P.127_80` vẫn không cải thiện nhiều, nguyên nhân có thể không nằm ở SVM mà nằm ở biểu diễn feature.

Ảnh 64x64 và PCA có thể làm mất chi tiết nhỏ như nét cong của số 6/8 hoặc biến dạng phối cảnh của biển cảnh báo. Khi đó, tăng `C`, đổi `gamma`, hoặc tăng class weight chỉ giúp trong giới hạn nhất định, nhưng không thể tạo lại thông tin đã mất trong feature.

## 4. Trade-off hiệu năng vs tốc độ triển khai

SVM kernel predict phụ thuộc vào số support vectors và số chiều feature:

`t_inference ∝ n_support_vectors × d`

Phase 1-New reference:

- Test Macro F1: `95.64%`
- Speed: `1.9318776733231922 ms/image`

Phase 2-New Best Tuned SVM:

- Test Macro F1: `96.29%`
- Speed: `1.5194648649701115 ms/image`
- Support vectors: `4514`

Nếu F1 tăng nhỏ nhưng số support vectors và inference time tăng mạnh, lựa chọn kỹ sư nên ưu tiên model cân bằng hơn cho deployment.

## 5. Memory optimization note

Mặc dù M06 yêu cầu huấn luyện nhiều cấu hình SVM, notebook không giữ toàn bộ model trong RAM. Với mỗi cấu hình, notebook chỉ lưu metric vào bảng kết quả. Model object chỉ được giữ lại nếu nó là best model hiện tại theo Validation Macro F1. Các model không tốt nhất được xóa ngay sau khi đánh giá bằng `del model` và `gc.collect()` để giảm áp lực RAM và tránh sập kernel khi chạy grid search.
