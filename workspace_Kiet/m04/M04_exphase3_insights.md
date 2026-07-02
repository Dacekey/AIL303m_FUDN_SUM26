# Báo cáo Insight M04 - PCA, Minority Augmentation và Final Evaluation

## 1. Mục tiêu

M04 là bước hoàn thiện pipeline cuối cùng của project phân loại biển báo. Nhiệm vụ chính là kiểm chứng giả thuyết rằng việc kết hợp PCA với Minority Augmentation có thể:

- Giảm số chiều đặc trưng để tăng tốc inference.
- Giữ hoặc cải thiện Macro F1 trên validation.
- Cải thiện Recall/F1 cho một số class thiểu số.
- Chọn một mô hình tốt nhất bằng validation.
- Mở test split đúng một lần để lấy kết quả final.

Notebook dùng cho báo cáo này:

```text
workspace_Kiet/m04/M04_final_evaluation_clean_from_start_ran.ipynb
```

## 2. Nguồn dữ liệu và output

Nguồn dữ liệu chính:

```text
paper/cropped_dataset/train
paper/cropped_dataset/val
paper/cropped_dataset/test
workspace_Kiet/support/minority_classes.csv
```

Output M04 đã được tổ chức lại theo folder:

```text
workspace_Kiet/m04_outputs
|-> cache
|   |-> m04_features_*.npz
|   |-> m04_augmented_features_*.npz
|   |-> m04_test_features_*.npz
|
|-> dataset
|   |-> m04_class_counts.csv
|   |-> m04_minority_classes.csv
|
|-> validation
|   |-> m04_validation_results_tn31_tn32_tn33.csv
|   |-> m04_minority_validation_report_by_experiment.csv
|   |-> m04_best_validation_model.joblib
|   |-> m04_best_validation_confusion_matrix.csv
|   |-> m04_best_validation_top_confusions.csv
|   |-> m04_final_pipeline_summary.json
|
|-> final_test
    |-> m04_final_test_metrics.csv
    |-> m04_final_test_classification_report.csv
    |-> m04_final_test_confusion_matrix.csv
    |-> m04_final_test_top_confusions.csv
    |-> m04_final_model.joblib
    |-> m04_final_test_summary.json
    |-> m04_output_checklist.csv
```

## 3. Cấu hình pipeline

Pipeline M04 sử dụng:

```text
Feature: HOG gray + Color Histogram
Image size: 64
Preprocess: pad_square
Scaler: StandardScaler
PCA: n_components = 0.95
Classifier: SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced')
Minority Augmentation: xoay nhẹ / chỉnh sáng trực tiếp trên ảnh train
```

Notebook không dùng SMOTE. Augmentation được thực hiện trên ảnh 2D trước khi trích xuất feature, phù hợp với yêu cầu mission M04.

## 4. Danh sách class thiểu số

M04 nhận danh sách class thiểu số từ M01, theo điều kiện `train < 30`.

| Class | Train | Val | Test |
| --- | ---: | ---: | ---: |
| R.301e | 16 | 2 | 3 |
| W.205c | 17 | 2 | 3 |
| S.505a_Xe máy | 20 | 2 | 3 |
| I.409 | 20 | 3 | 3 |
| S.505a_Xe tải và công | 21 | 3 | 3 |
| W.239b_ | 21 | 3 | 3 |
| W.203c | 23 | 3 | 3 |
| P.124d | 23 | 3 | 3 |
| W.205a | 26 | 3 | 4 |
| W.225 | 28 | 4 | 4 |

Nhóm này được đưa vào pipeline Minority Augmentation trong cấu hình TN3.3.

## 5. Kết quả validation

Ba thí nghiệm chính của M04:

| Mã TN | Cấu hình | PCA | Train samples | Dim trước PCA | Dim sau PCA | Val Accuracy | Val Macro F1 | Inference ms/ảnh |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TN3.1 | No PCA, No Augmentation | Không | 6605 | 1812 | 1812 | 96.24% | 94.85% | 5.93 |
| TN3.2 | PCA 95%, No Augmentation | Có | 6605 | 1812 | 569 | 96.60% | 95.04% | 1.91 |
| TN3.3 | PCA 95% + Minority Augmentation | Có | 7035 | 1812 | 571 | 96.72% | 95.77% | 1.96 |

Kết quả cho thấy:

- PCA giảm số chiều từ `1812` xuống khoảng `569-571`.
- TN3.2 nhanh hơn TN3.1 rõ rệt: khoảng `1.91 ms/ảnh` so với `5.93 ms/ảnh`.
- TN3.3 có Val Macro F1 cao nhất: `95.77%`.
- Minority Augmentation tăng số mẫu train từ `6605` lên `7035`.
- Best validation experiment là `TN3.3`.

## 6. Phân tích tác động của PCA

So sánh TN3.1 và TN3.2:

| So sánh | TN3.1 | TN3.2 | Nhận xét |
| --- | ---: | ---: | --- |
| Feature dimension | 1812 | 569 | PCA giảm khoảng 68.6% số chiều |
| Val Macro F1 | 94.85% | 95.04% | Không giảm, còn tăng nhẹ |
| Inference ms/ảnh | 5.93 | 1.91 | Tăng tốc khoảng 3.1 lần |

Kết luận: PCA 95% đạt đúng mục tiêu giảm chiều và tăng tốc inference mà vẫn giữ chất lượng mô hình. Đây là bằng chứng tốt cho giả thuyết PCA trong M04.

## 7. Phân tích Minority Augmentation

So sánh TN3.2 và TN3.3:

| So sánh | TN3.2 | TN3.3 | Nhận xét |
| --- | ---: | ---: | --- |
| Train samples | 6605 | 7035 | Tăng mẫu cho class thiểu số |
| Val Macro Recall | 94.08% | 95.24% | Cải thiện recall tổng thể |
| Val Macro F1 | 95.04% | 95.77% | TN3.3 tốt nhất |
| Inference ms/ảnh | 1.91 | 1.96 | Gần như không tăng đáng kể |

Một số class thiểu số cải thiện rõ trên validation:

| Class | TN3.1 Recall/F1 | TN3.2 Recall/F1 | TN3.3 Recall/F1 | Nhận xét |
| --- | --- | --- | --- | --- |
| W.205a | 66.67% / 80.00% | 66.67% / 80.00% | 100.00% / 100.00% | Cải thiện rõ sau augmentation |
| W.225 | 75.00% / 85.71% | 75.00% / 85.71% | 100.00% / 100.00% | Cải thiện rõ sau augmentation |
| W.205c | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | Vẫn là class khó, cần theo dõi thêm |

Nhận xét: Minority Augmentation giúp cải thiện Macro F1 và một số class ít mẫu, nhưng không giải quyết toàn bộ vấn đề. Riêng `W.205c` vẫn không được nhận diện tốt trên validation, cho thấy một số class cần thêm dữ liệu thật hoặc kiểm tra lại chất lượng ảnh/nhãn.

## 8. Best model

Mô hình được chọn theo validation:

```text
Best experiment: TN3.3
Configuration: PCA 95% + Minority Augmentation
Val Accuracy: 96.72%
Val Macro F1: 95.77%
```

Artifact model validation:

```text
workspace_Kiet/m04_outputs/validation/m04_best_validation_model.joblib
```

Sau khi chốt TN3.3 bằng validation, notebook mở test split và chạy final evaluation đúng một lần.

## 9. Final test result

Kết quả final trên test split:

| Metric | Giá trị |
| --- | ---: |
| Best experiment | TN3.3 - PCA 95% + Minority Augmentation |
| Test samples | 851 |
| Test Accuracy | 96.24% |
| Test Macro Precision | 98.08% |
| Test Macro Recall | 94.39% |
| Test Macro F1 | 95.64% |
| Test Weighted F1 | 96.15% |
| Test inference ms/ảnh | 1.93 |

Final model:

```text
workspace_Kiet/m04_outputs/final_test/m04_final_model.joblib
```

Final report files:

```text
workspace_Kiet/m04_outputs/final_test/m04_final_test_metrics.csv
workspace_Kiet/m04_outputs/final_test/m04_final_test_classification_report.csv
workspace_Kiet/m04_outputs/final_test/m04_final_test_confusion_matrix.csv
workspace_Kiet/m04_outputs/final_test/m04_final_test_top_confusions.csv
workspace_Kiet/m04_outputs/final_test/m04_final_test_summary.json
```

## 10. Final test confusion

Các cặp nhầm nhiều nhất trên test:

| Class thật | Class dự đoán | Số lần nhầm | Nhận xét |
| --- | --- | ---: | --- |
| P.127_80 | P.127_60 | 4 | Nhầm chữ số trong nhóm biển tốc độ |
| W.224 | W.245a | 3 | Nhầm nhóm biển cảnh báo |
| P.131a | P.130 | 3 | Hai biển giống nhau về ký hiệu/hình dạng |
| P.130 | P.131a | 3 | Nhầm hai chiều P.130/P.131a |
| W.207 | W.224 | 3 | Nhầm nhóm biển cảnh báo |
| W.205c | W.245a | 1 | Class thiểu số vẫn khó |
| W.239b_ | P.130 | 1 | Class ít mẫu, dễ bị hút về class lớn |
| W.239b_ | W.221b | 1 | Class ít mẫu, cần theo dõi thêm |

Nhận xét: Sau khi dùng PCA + Minority Augmentation, model đạt điểm tổng thể cao, nhưng các lỗi còn lại vẫn tập trung vào nhóm class có ký hiệu gần nhau hoặc cần đọc chi tiết nhỏ như chữ số tốc độ.

## 11. Đánh giá mức độ hoàn thành M04

| Yêu cầu M04 | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Nhận danh sách class thiểu số từ M01 | Hoàn thành | `m04_outputs/dataset/m04_minority_classes.csv` |
| Không dùng SMOTE | Hoàn thành | Augmentation trực tiếp trên ảnh |
| Chạy TN3.1 baseline | Hoàn thành | `m04_validation_results_tn31_tn32_tn33.csv` |
| Chạy TN3.2 PCA 95% | Hoàn thành | Dim giảm từ 1812 xuống 569 |
| Chạy TN3.3 PCA + Minority Augmentation | Hoàn thành | Train samples tăng lên 7035, Macro F1 cao nhất |
| Chọn best model bằng validation | Hoàn thành | TN3.3 được chọn |
| Chạy final test đúng một lần | Hoàn thành | `m04_final_test_metrics.csv` |
| Lưu final model | Hoàn thành | `m04_final_model.joblib` |
| Xuất report final | Hoàn thành | metrics, classification report, confusion matrix, top confusions, summary |

## 12. Kết luận

M04 đã hoàn thành đúng yêu cầu mission. Kết quả validation chứng minh PCA giúp giảm mạnh số chiều và tăng tốc inference mà không làm giảm Macro F1. Khi kết hợp thêm Minority Augmentation, TN3.3 đạt kết quả validation tốt nhất với Val Macro F1 `95.77%`.

Sau khi chốt TN3.3 bằng validation, notebook đã chạy final test trên 851 ảnh test và đạt Test Macro F1 `95.64%`, Test Accuracy `96.24%`. Đây là kết quả final chính thức của M04.

Kết luận cuối cùng:

```text
Final model: TN3.3 - PCA 95% + Minority Augmentation
Feature: HOG gray + Color Histogram
Preprocess: pad_square
Test Accuracy: 96.24%
Test Macro F1: 95.64%
Inference: 1.93 ms/ảnh
```

Mô hình đã đạt mục tiêu cân bằng giữa chất lượng nhận diện và tốc độ suy luận. Các lỗi còn lại chủ yếu nằm ở nhóm biển có ký hiệu giống nhau hoặc chữ số nhỏ, đặc biệt nhóm `P.127_xx`, `P.130/P.131a`, và một số biển cảnh báo.
