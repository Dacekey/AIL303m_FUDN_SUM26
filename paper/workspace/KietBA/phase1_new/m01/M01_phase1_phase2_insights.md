# Báo cáo Insight M01 - Error Analysis và Minority Diagnosis

## 1. Mục tiêu

M01 có nhiệm vụ kiểm chứng lại các lỗi quan trọng từ kết quả Phase 1 và Phase 2 trước khi nhóm bước sang phần tối ưu cuối. Trọng tâm của phân tích là:

- Xác định các class thiểu số có số lượng ảnh train thấp, đặc biệt nhóm `train < 30`.
- Tìm các class có Recall hoặc F1 thấp trên validation.
- Phân tích các cặp class bị nhầm nhiều nhất để hỗ trợ Member 2 kiểm chứng preprocessing/resize.
- Bàn giao danh sách class cần ưu tiên cho Member 4 khi làm Minority Augmentation trong M04.

Notebook dùng cho báo cáo này:

```text
workspace_Kiet/m01/M01_phase1_phase2_error_analysis_FIXED.ipynb
```

## 2. Nguồn dữ liệu

Các kết quả được tổng hợp từ dữ liệu gốc của project:

```text
paper/result/exphase_1_result
paper/result/exphase_2_result
paper/cropped_dataset
workspace_Kiet/support
```

Các file hỗ trợ được notebook đọc lại:

```text
workspace_Kiet/support/m01_phase12_bottleneck_phase1.csv
workspace_Kiet/support/m01_phase12_bottleneck_phase2.csv
workspace_Kiet/support/m01_phase12_top_confusions.csv
workspace_Kiet/support/m01_phase12_confusion_groups.csv
workspace_Kiet/support/m01_phase12_handover_m2.csv
workspace_Kiet/support/m01_phase12_handover_m4.csv
```

## 3. Tóm tắt kết quả chính

| Hạng mục | Kết quả | Nhận xét |
| --- | ---: | --- |
| Best Phase 1 | Raw Pixels (Baseline) | Val Macro F1 93.40%, nhưng không có per-class validation report |
| Best Phase 2 | Edge Histogram | Val Macro F1 58.75%, có per-class validation report |
| Class thiểu số | 10 | Các class có `train < 30` |
| Bottleneck Phase 2 | 28 | Class có Recall/F1 < 60% hoặc train support < 30 |
| Kết luận chính | Cần dùng cả Phase 1 và Phase 2 | Phase 1 mạnh về tổng quan, Phase 2 mạnh về chẩn đoán theo class |

Phase 1 có điểm tổng thể cao, nhưng vì chưa lưu per-class validation report nên không thể dùng Phase 1 để kết luận trực tiếp Recall/F1 của từng class. Phase 2 có điểm tổng thể thấp hơn, nhưng có `val_classification_report_edge_histogram.csv`, vì vậy phù hợp hơn để xác định bottleneck class theo Recall/F1 thật.

## 4. Phase 1 Summary

| Cấu hình | Đặc trưng | Val Accuracy | Val Macro Recall | Val Macro F1 | Ghi chú |
| --- | --- | ---: | ---: | ---: | --- |
| Raw Pixels (Baseline) | Raw Pixels (gray) | 94.30% | 91.45% | 93.40% | Best theo Macro F1 |
| HOG + Color Histogram - gray | HOG gray + HSV histogram | 95.51% | 91.40% | 93.27% | Accuracy cao nhất, Macro F1 sát best |
| HOG only - gray | HOG gray | 95.27% | 91.18% | 92.86% | Ổn định |
| HOG only - yuv | HOG YUV | 95.51% | 91.17% | 92.74% | Accuracy cao nhưng Macro F1 thấp hơn |
| HOG + Color Histogram - yuv | HOG YUV + HSV histogram | 95.51% | 90.18% | 92.09% | Không vượt baseline |

Nhận xét: Phase 1 cho thấy bài toán có thể đạt hiệu năng cao với feature đơn giản hoặc HOG/Color Histogram. Tuy nhiên, thiếu per-class report làm hạn chế khả năng phân tích lỗi chi tiết. Vì vậy Phase 1 được dùng chủ yếu để lấy baseline tổng quan, top confusion và danh sách class thiếu mẫu.

## 5. Phase 2 Summary

| Cấu hình | Đặc trưng | Val Accuracy | Val Macro Recall | Val Macro F1 | Ghi chú |
| --- | --- | ---: | ---: | ---: | --- |
| Edge Histogram | Edge Histogram | 64.20% | 60.95% | 58.75% | Best Phase 2, có per-class report |
| Gabor | Gabor | 59.34% | 69.95% | 58.74% | Macro Recall cao nhưng F1 không vượt Edge |
| LBP | LBP | 37.99% | 37.20% | 30.74% | Kém ổn định |
| Hu Moments | Hu Moments | 31.43% | 36.57% | 25.11% | Không phù hợp làm feature chính |

Nhận xét: Phase 2 không tốt bằng Phase 1 về điểm tổng thể, nhưng lại rất hữu ích cho M01 vì có báo cáo từng class. Edge Histogram là cấu hình tốt nhất của Phase 2 và được dùng làm nguồn chính để phân tích bottleneck.

## 6. Bottleneck Phase 1

Do Phase 1 không có per-class validation report, bảng bottleneck Phase 1 dùng điều kiện `train < 30` để nhận diện class thiểu số.

| Class ID | Train | Val Support | Recall Val | F1 Val | Chẩn đoán |
| --- | ---: | ---: | --- | --- | --- |
| R.301e | 16 | 2 | N/A | N/A | Thiếu mẫu train nghiêm trọng |
| W.205c | 17 | 2 | N/A | N/A | Thiếu mẫu train nghiêm trọng |
| I.409 | 20 | 3 | N/A | N/A | Thiếu mẫu train |
| S.505a_Xe máy | 20 | 2 | N/A | N/A | Thiếu mẫu train |
| S.505a_Xe tải và công | 21 | 3 | N/A | N/A | Thiếu mẫu train |
| W.239b_ | 21 | 3 | N/A | N/A | Thiếu mẫu train |
| P.124d | 23 | 3 | N/A | N/A | Thiếu mẫu train |
| W.203c | 23 | 3 | N/A | N/A | Thiếu mẫu train |
| W.205a | 26 | 3 | N/A | N/A | Thiếu mẫu train |
| W.225 | 28 | 4 | N/A | N/A | Thiếu mẫu train |

Đây là danh sách minority class chính thức để bàn giao cho M04 nếu chỉ xét tiêu chí số lượng mẫu.

## 7. Bottleneck Phase 2

Điều kiện lọc Phase 2:

```text
Recall Val < 60% hoặc F1 Val < 60% hoặc Train < 30
```

Các class yếu nhất cần chú ý:

| Class ID | Train | Val Support | Recall Val | F1 Val | Chẩn đoán |
| --- | ---: | ---: | ---: | ---: | --- |
| W.205a | 26 | 3 | 0.00% | 0.00% | Thiếu mẫu, Recall/F1 rất thấp |
| W.205c | 17 | 2 | 0.00% | 0.00% | Thiếu mẫu, Recall/F1 rất thấp |
| P.124d | 23 | 3 | 33.33% | 25.00% | Thiếu mẫu, F1 thấp |
| W.246a | 64 | 8 | 25.00% | 25.00% | Recall/F1 thấp dù không thuộc nhóm `train < 30` |
| P.107 | 171 | 21 | 28.57% | 33.33% | Có dữ liệu nhưng nhận diện kém |
| W.203c | 23 | 3 | 33.33% | 33.33% | Thiếu mẫu, Recall/F1 thấp |
| I.409 | 20 | 3 | 33.33% | 40.00% | Thiếu mẫu, Recall/F1 thấp |
| P.127_80 | 168 | 21 | 38.10% | 42.11% | Nhóm biển tốc độ dễ nhầm |
| W.207 | 204 | 26 | 38.46% | 42.55% | Nhóm biển cảnh báo dễ nhầm |
| P.104 | 36 | 4 | 50.00% | 44.44% | Support validation thấp |

Nhận xét: Bottleneck không chỉ đến từ thiếu dữ liệu. Một số class có nhiều ảnh train như `P.107`, `P.127_80`, `W.207`, `W.224`, `P.131a`, `P.102` vẫn có F1/Recall thấp, cho thấy lỗi còn đến từ hình dạng, độ tương đồng ký hiệu, resize/preprocessing hoặc đặc trưng chưa phân biệt đủ tốt.

## 8. Top Confusion Analysis

Các cặp nhầm nổi bật:

| Phase | Class thật | Class dự đoán | Số lần nhầm | Nhận xét |
| --- | --- | --- | ---: | --- |
| Phase 1 | P.130 | P.131a | 8 | Hai biển giống nhau về hình dạng/ký hiệu |
| Phase 1 | W.224 | P.130 | 5 | Nhầm nhóm cảnh báo/cấm |
| Phase 1 | P.131a | P.130 | 4 | Nhầm hai chiều P.130/P.131a |
| Phase 1 | P.127_80 | P.127_60 | 3 | Nhầm chữ số trong biển tốc độ |
| Phase 2 | P.130 | P.102 | 17 | Bị hút về class lớn/dễ nhận diện |
| Phase 2 | P.131a | P.130 | 14 | Nhầm biển giống nhau |
| Phase 2 | R.302a | P.102 | 13 | Nhầm với class P.102 |
| Phase 2 | P.127_60 | P.127_50 | 4 | Nhầm chữ số tốc độ |
| Phase 2 | P.127_60 | P.127_80 | 4 | Nhầm chữ số tốc độ |
| Phase 2 | P.127_40 | P.127_50 | 4 | Nhầm chữ số tốc độ |

Các nhóm lỗi chính:

| Nhóm lỗi | Class liên quan | Bằng chứng |
| --- | --- | --- |
| Nhầm biển giống nhau | P.130, P.131a | Cả Phase 1 và Phase 2 đều có P.130/P.131a trong top confusion |
| Nhầm biển tốc độ | P.127_40, P.127_50, P.127_60, P.127_80 | Các biển khác nhau chủ yếu ở chữ số, dễ mất chi tiết khi resize |
| Nhầm với class lớn | P.102, P.130, R.302a, P.124c, P.131a | Phase 2 có nhiều class bị dự đoán về P.102/P.130 |
| Nhầm biển cảnh báo | W.207, W.224, W.245a | Nhóm biển cảnh báo có hình học/ký hiệu gần nhau |

## 9. Handover cho Member 2

Member 2 nên dùng các nhóm sau để kiểm chứng giả thuyết về preprocessing, đặc biệt là `stretch` so với `pad_square`.

| Nhóm/Class | Lý do | Việc cần làm |
| --- | --- | --- |
| Biển không vuông | Resize trực tiếp về 64x64 có thể làm méo hình học | So sánh `stretch` với `pad_square` |
| P.127_40/50/60/80 | Nhầm chữ số tốc độ | Kiểm tra image size, resize và độ rõ chữ số |
| P.130/P.131a và nhóm P.102 | Nhóm class bị hút nhầm nhiều | Kiểm tra feature có đủ phân biệt ký hiệu nhỏ không |
| W.207/W.224/W.245a | Nhóm biển cảnh báo dễ nhầm | Kiểm tra bảo toàn hình dạng tam giác/cảnh báo |

## 10. Handover cho Member 4

Danh sách class thiểu số chính thức để ưu tiên Minority Augmentation:

```text
R.301e
W.205c
I.409
S.505a_Xe máy
S.505a_Xe tải và công
W.239b_
P.124d
W.203c
W.205a
W.225
```

Ngoài danh sách 10 class trên, Member 4 nên theo dõi thêm các class có Recall/F1 thấp dù không quá ít mẫu, ví dụ:

```text
W.246a, P.107, P.127_80, W.207, P.104, P.129, W.224, P.131a, P.117_, P.102, P.124c
```

Lý do là augmentation chỉ giải quyết một phần vấn đề thiếu mẫu. Các class nhiều mẫu nhưng vẫn bị nhầm cần được đánh giá bằng confusion matrix, preprocessing và feature extraction.

## 11. Đánh giá mức độ hoàn thành M01

| Yêu cầu M01 | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Đọc và đối chiếu Phase 1 | Hoàn thành | Có summary Phase 1 và top confusion Phase 1 |
| Đọc và đối chiếu Phase 2 | Hoàn thành | Có `feature_svm_phase2_results` và per-class report |
| Xác định class thiểu số | Hoàn thành | 10 class `train < 30` |
| Xác định bottleneck class | Hoàn thành | 28 class theo Recall/F1/support |
| Phân tích top confusion | Hoàn thành | Có nhóm lỗi P.130/P.131a, P.127_xx, P.102/P.130, W.xxx |
| Bàn giao cho Member 2 | Hoàn thành | Có nhóm lỗi liên quan preprocessing/resize |
| Bàn giao cho Member 4 | Hoàn thành | Có danh sách class cho Minority Augmentation |

## 12. Kết luận

M01 đã đạt yêu cầu của mission. Kết quả cho thấy Phase 1 có hiệu năng tổng thể cao nhưng chưa đủ thông tin để chẩn đoán lỗi theo class. Phase 2 bổ sung per-class validation report, giúp xác định rõ các bottleneck class và các nhóm nhầm lẫn quan trọng.

Kết luận quan trọng nhất là lỗi không chỉ đến từ thiếu dữ liệu. Ngoài 10 class thiểu số cần augmentation, còn nhiều nhóm class có đủ dữ liệu nhưng vẫn nhầm do hình dạng, ký hiệu giống nhau hoặc khả năng mất chi tiết khi resize. Vì vậy, handover cho M04 cần bao gồm cả danh sách minority class và danh sách class cần theo dõi Recall/F1 sau augmentation.
