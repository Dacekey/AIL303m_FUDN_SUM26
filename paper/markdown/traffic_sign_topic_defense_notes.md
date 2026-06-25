# Defense Notes: Why This Traffic Sign Feature Extraction Topic Is Still Valid

## 0. Purpose of this document

This document helps explain and defend the research direction:

**A Comparative Study of Handcrafted Feature Extraction Methods for Vietnamese Traffic Sign Classification Using Traditional Machine Learning**

Vietnamese title:

**So sánh các phương pháp trích xuất đặc trưng thủ công cho phân loại biển báo giao thông Việt Nam bằng học máy truyền thống**

The goal is to prepare answers for questions such as:

- If this topic already exists, what is different?
- Why is using a Vietnamese traffic sign dataset meaningful?
- What is the value of the research if the algorithms are not new?
- Why focus on feature extraction instead of full detection?
- Why is this topic still enough for a course-level research paper?

## 1. Core defense idea

The project should not claim to propose a completely new algorithm.

The correct positioning is:

> This study is an empirical comparison of handcrafted feature extraction methods for Vietnamese traffic sign classification using traditional Machine Learning.

Vietnamese version:

> Nghiên cứu này là một so sánh thực nghiệm các phương pháp trích xuất đặc trưng thủ công cho bài toán phân loại biển báo giao thông Việt Nam bằng học máy truyền thống.

The novelty is not algorithmic novelty.

The value comes from:

1. Applying and evaluating traditional feature extraction methods on Vietnamese traffic sign data.
2. Comparing features in a controlled experimental setup.
3. Understanding which handcrafted feature works better in the Vietnamese traffic sign domain.
4. Providing a traditional Machine Learning baseline without using Deep Learning.
5. Analyzing errors and confusion between traffic sign classes in the Vietnamese context.

## 2. If the teacher asks: "This direction already exists, so what is different?"

Recommended answer:

> Yes, handcrafted feature extraction for traffic sign recognition has been studied before, especially on international benchmarks such as GTSRB and GTSDB. However, our project does not claim to introduce a new feature extraction algorithm. Instead, we conduct a controlled empirical comparison on a Vietnamese traffic sign dataset. The purpose is to verify whether common handcrafted features such as Color Histogram, LBP, HOG, and HOG + Color remain effective under Vietnamese traffic sign data, which may differ in sign design, class distribution, image quality, and real-world context.

Vietnamese version:

> Dạ đúng là hướng trích xuất đặc trưng thủ công cho nhận dạng biển báo giao thông đã có nhiều nghiên cứu trước, đặc biệt trên các benchmark quốc tế như GTSRB/GTSDB. Tuy nhiên, nhóm em không claim thuật toán mới. Nhóm em thực hiện một nghiên cứu thực nghiệm có kiểm soát trên dữ liệu biển báo giao thông Việt Nam để kiểm chứng xem các đặc trưng như Color Histogram, LBP, HOG và HOG + Color có còn hiệu quả trong bối cảnh dữ liệu Việt Nam hay không.

Short version:

> The difference is not the algorithm itself, but the dataset domain, controlled comparison, and analysis in the Vietnamese traffic sign context.

Vietnamese version:

> Điểm khác biệt không nằm ở thuật toán, mà nằm ở miền dữ liệu, thiết kế thí nghiệm có kiểm soát, và phân tích kết quả trên bối cảnh biển báo Việt Nam.

## 3. Why using a Vietnamese dataset is meaningful

### 3.1 Traffic signs in Vietnam may differ from international benchmarks

Many existing studies use datasets such as GTSRB, which contains German traffic signs. However, Vietnamese traffic signs may differ in:

- sign symbols,
- sign codes,
- class categories,
- road environment,
- frequency of different sign types,
- text and number presentation,
- visual design details,
- traffic context.

Even if the general sign shapes are similar, handcrafted features are sensitive to visual details.

For example:

- HOG depends on edge and shape distribution.
- LBP depends on local texture patterns.
- Color Histogram depends on color distribution.
- Raw Pixel features depend directly on visual appearance.

Therefore, results from GTSRB do not automatically transfer to Vietnamese traffic signs.

### 3.2 Vietnamese road scenes have different visual conditions

Vietnamese traffic images can contain:

- many motorcycles,
- complex road backgrounds,
- trees, electric wires, advertisements, and buildings,
- small traffic signs,
- faded or dirty signs,
- occluded signs,
- unstable dashcam viewpoints,
- strong sunlight or backlight,
- blur from motion,
- weather and lighting variation.

These factors affect feature extraction.

For example:

| Feature | What it relies on | Possible weakness in Vietnamese data |
|---|---|---|
| Color Histogram | color distribution | lighting change, faded signs, similar colors |
| LBP | local texture | blur, noise, low resolution |
| HOG | edge and shape | small signs, occlusion, inaccurate crop |
| HOG + Color | shape and color | higher feature dimension, needs scaling |

Thus, testing on Vietnamese traffic sign data has practical value.

### 3.3 The result has local application value

If the target application is traffic sign recognition in Vietnam, evaluating only on GTSRB is not enough.

A Vietnamese dataset helps answer a more relevant question:

> If Deep Learning is not allowed, which handcrafted feature is most suitable for Vietnamese traffic sign classification?

Vietnamese version:

> Nếu không dùng Deep Learning, đặc trưng thủ công nào phù hợp nhất cho phân loại biển báo giao thông Việt Nam?

This is a small but valid research question for a course-level paper.

## 4. Difference between previous studies and this project

| Aspect | Previous studies | This project |
|---|---|---|
| Main domain | Often GTSRB/GTSDB or other international datasets | Vietnamese traffic sign dataset |
| Main goal | Traffic sign recognition in general | Vietnamese traffic sign classification |
| Method type | Handcrafted features or classifier comparison | Controlled feature extraction comparison |
| Feature set | HOG, SIFT, LBP, Gabor, BoW depending on paper | Raw Pixel, Color Histogram, LBP, HOG, HOG + Color |
| Classifier | Often multiple classifiers | Mainly fixed SVM for fair feature comparison |
| Detection | Some studies include full detection | Classification from bounding-box crops |
| Novelty claim | Sometimes method improvement | Empirical validation in Vietnamese domain |
| Scope | May be broader | Course-level, focused, reproducible |

Key message:

> Our project is not a new algorithm paper. It is a domain-specific experimental comparison paper.

Vietnamese version:

> Bài của nhóm không phải là bài đề xuất thuật toán mới, mà là bài so sánh thực nghiệm theo miền dữ liệu cụ thể.

## 5. Why the dataset domain can change the result

The same algorithm can perform differently on different datasets due to:

1. Different number of classes.
2. Different class imbalance.
3. Different sign designs.
4. Different image quality.
5. Different object sizes.
6. Different crop quality.
7. Different backgrounds.
8. Different lighting and weather conditions.
9. Different annotation quality.

Therefore, even if HOG + SVM works well on GTSRB, it does not prove that it will be the best method on Vietnamese traffic signs.

A valid research objective is:

> To verify whether conclusions from international traffic sign datasets still hold for Vietnamese traffic sign classification.

Vietnamese version:

> Kiểm chứng xem các kết luận từ các bộ dữ liệu biển báo quốc tế có còn đúng trên dữ liệu biển báo giao thông Việt Nam hay không.

## 6. Why focusing on feature extraction is still valid

The teacher suggested focusing on feature extraction. This project follows that direction.

The main design is:

```text
Use the same dataset
Use the same train/test split
Use the same classifier
Only change feature extraction method
Compare performance
```

This makes the experiment controlled.

If both feature and classifier are changed at the same time, it becomes difficult to know whether the result improves because of the feature or because of the classifier.

Therefore, the project should fix the classifier, for example:

```text
SVM classifier
```

Then compare:

```text
Raw Pixel + SVM
Color Histogram + SVM
LBP + SVM
HOG + SVM
HOG + Color Histogram + SVM
```

This directly answers the feature extraction question.

## 7. If the teacher asks: "Why not use GTSRB?"

Recommended answer:

> GTSRB is a strong benchmark and useful for related work. However, the goal of this project is Vietnamese traffic sign classification. If we only use GTSRB, the result is easier to compare with existing papers but less relevant to the Vietnamese context. Using a Vietnamese dataset allows us to evaluate whether handcrafted features behave similarly under Vietnamese traffic sign images.

Vietnamese version:

> GTSRB là benchmark tốt và nhóm em có thể dùng để tham khảo trong related work. Tuy nhiên, mục tiêu của nhóm là phân loại biển báo giao thông Việt Nam. Nếu chỉ dùng GTSRB thì kết quả dễ so sánh với paper trước, nhưng ít liên quan đến bối cảnh Việt Nam. Dùng dataset Việt Nam giúp kiểm tra xem các đặc trưng thủ công có còn hiệu quả trong điều kiện dữ liệu Việt Nam hay không.

Optional addition:

> If time allows, GTSRB can be used as an external reference, but the main dataset should remain Vietnamese traffic signs.

Vietnamese version:

> Nếu còn thời gian, nhóm em có thể dùng GTSRB như một bộ dữ liệu tham chiếu phụ, nhưng dataset chính vẫn nên là biển báo Việt Nam.

## 8. If the teacher asks: "Why not do full detection?"

Recommended answer:

> A complete traffic sign recognition system includes both detection and classification. However, this project focuses on the classification stage because the main research objective is feature extraction. We use available bounding box annotations to crop the traffic sign regions. This avoids mixing detection errors with classification errors and allows a fair comparison between feature extraction methods.

Vietnamese version:

> Một hệ thống nhận dạng biển báo hoàn chỉnh gồm cả phát hiện vị trí và phân loại. Tuy nhiên, bài của nhóm tập trung vào giai đoạn phân loại vì mục tiêu chính là so sánh feature extraction. Nhóm dùng bounding box annotation có sẵn để crop vùng biển báo, nhờ đó tránh việc lỗi detection làm nhiễu kết quả classification và giúp so sánh công bằng hơn giữa các phương pháp trích xuất đặc trưng.

Key phrase:

> We isolate the classification problem to evaluate feature extraction more fairly.

Vietnamese version:

> Nhóm cô lập bài toán classification để đánh giá feature extraction công bằng hơn.

## 9. If the teacher asks: "What is the practical value?"

Recommended answer:

> The practical value is that the study identifies which traditional handcrafted feature is most suitable when Deep Learning is not allowed or when computational resources are limited. The result can serve as a baseline for Vietnamese traffic sign classification.

Vietnamese version:

> Giá trị thực tế của bài là xác định đặc trưng thủ công nào phù hợp hơn khi không dùng Deep Learning hoặc khi tài nguyên tính toán hạn chế. Kết quả có thể dùng như một baseline truyền thống cho bài toán phân loại biển báo giao thông Việt Nam.

Possible conclusions the project may produce:

- HOG may outperform Raw Pixel because it captures shape and gradient structure.
- Color Histogram may help distinguish signs with different dominant colors.
- LBP may be lightweight but weaker for blurred or small signs.
- HOG + Color may perform better because it combines shape and color.
- Some classes may be confused because their symbols or shapes are visually similar.

## 10. If the teacher asks: "Is this enough for a research paper?"

Recommended answer:

> For a course-level research paper, yes. The project has a clear research question, related work, dataset, experimental design, metrics, result comparison, and error analysis. The paper does not need to claim state-of-the-art performance. It only needs to provide a well-structured empirical comparison and discussion.

Vietnamese version:

> Với mức độ bài nghiên cứu môn học thì đủ. Bài có câu hỏi nghiên cứu rõ ràng, có related work, có dataset, có thiết kế thí nghiệm, có metric đánh giá, có bảng so sánh kết quả và phân tích lỗi. Nhóm không cần claim state-of-the-art, chỉ cần trình bày một so sánh thực nghiệm rõ ràng và có phân tích.

Minimum elements required:

1. Clear research question.
2. Clear dataset description.
3. Controlled experiment.
4. Multiple feature extraction methods.
5. Traditional ML classifier.
6. Evaluation metrics.
7. Confusion matrix.
8. Error analysis.
9. Limitations.
10. Future work.

## 11. How to describe the novelty safely

Do not say:

> We propose a new traffic sign recognition method.

Instead say:

> We conduct a domain-specific empirical comparison of handcrafted feature extraction methods for Vietnamese traffic sign classification.

Vietnamese version:

> Nhóm thực hiện một so sánh thực nghiệm theo miền dữ liệu cụ thể đối với các phương pháp trích xuất đặc trưng thủ công cho phân loại biển báo giao thông Việt Nam.

Safe novelty statement:

> The novelty of this work lies not in a new algorithm, but in the controlled evaluation and analysis of traditional feature extraction methods on Vietnamese traffic sign data.

Vietnamese version:

> Điểm mới của nghiên cứu không nằm ở thuật toán mới, mà nằm ở việc đánh giá và phân tích có kiểm soát các phương pháp trích xuất đặc trưng truyền thống trên dữ liệu biển báo giao thông Việt Nam.

## 12. Suggested answer to give the teacher

This is a complete answer that can be reused:

> Dạ đúng là hướng handcrafted feature extraction cho traffic sign recognition đã có nhiều nghiên cứu trước, đặc biệt trên các benchmark như GTSRB. Tuy nhiên, nhóm em không claim thuật toán mới. Nhóm em định vị bài này là một comparative empirical study trên dữ liệu biển báo giao thông Việt Nam. Lý do là đặc điểm biển báo, chất lượng ảnh, bối cảnh đường phố và phân bố class ở Việt Nam có thể khác với benchmark nước ngoài. Vì vậy, kết quả từ GTSRB không đảm bảo sẽ chuyển nguyên sang dữ liệu Việt Nam. Nhóm em muốn kiểm chứng xem trong điều kiện không dùng Deep Learning, các đặc trưng như Color Histogram, LBP, HOG và HOG + Color hoạt động như thế nào, feature nào phù hợp nhất, và class nào dễ bị nhầm. Bài sẽ tập trung vào classification từ bounding box có sẵn để đánh giá công bằng tác động của feature extraction.

English version:

> It is true that handcrafted feature extraction for traffic sign recognition has been studied before, especially on benchmarks such as GTSRB. However, our work does not claim to introduce a new algorithm. We position this project as a comparative empirical study on Vietnamese traffic sign data. The reason is that sign design, image quality, road context, and class distribution in Vietnam may differ from international benchmarks. Therefore, results from GTSRB do not necessarily transfer directly to Vietnamese data. We aim to evaluate how traditional features such as Color Histogram, LBP, HOG, and HOG + Color perform when Deep Learning is not used, identify which feature is most suitable, and analyze which classes are commonly confused. The study focuses on classification from available bounding boxes to fairly evaluate the effect of feature extraction.

## 13. Recommended research framing in the paper

Use this framing in the Introduction:

> Traffic sign recognition has been widely studied using both traditional Machine Learning and Deep Learning approaches. However, most benchmark studies are conducted on international datasets such as GTSRB. In this study, we focus on Vietnamese traffic sign classification and evaluate handcrafted feature extraction methods under a traditional Machine Learning framework. By using bounding box annotations to crop traffic sign regions, the study isolates the classification stage and compares the effect of different feature representations.

Vietnamese version:

> Nhận dạng biển báo giao thông đã được nghiên cứu rộng rãi bằng cả học máy truyền thống và học sâu. Tuy nhiên, nhiều nghiên cứu benchmark được thực hiện trên các bộ dữ liệu quốc tế như GTSRB. Trong nghiên cứu này, nhóm tập trung vào phân loại biển báo giao thông Việt Nam và đánh giá các phương pháp trích xuất đặc trưng thủ công trong khuôn khổ học máy truyền thống. Bằng cách sử dụng bounding box annotation để crop vùng biển báo, nghiên cứu cô lập giai đoạn phân loại và so sánh ảnh hưởng của các biểu diễn đặc trưng khác nhau.

## 14. Recommended limitation statement

Use this in the Discussion or Conclusion:

> This study focuses only on the classification stage and uses ground-truth bounding boxes to crop traffic sign regions. Therefore, the reported results do not represent the performance of a complete end-to-end detection and recognition system. Future work can integrate a traditional detection pipeline or compare the handcrafted-feature baseline with modern Deep Learning detectors.

Vietnamese version:

> Nghiên cứu này chỉ tập trung vào giai đoạn phân loại và sử dụng bounding box có sẵn để crop vùng biển báo. Vì vậy, kết quả báo cáo chưa phản ánh hiệu năng của một hệ thống phát hiện và nhận dạng end-to-end hoàn chỉnh. Trong tương lai, có thể tích hợp thêm pipeline phát hiện truyền thống hoặc so sánh baseline đặc trưng thủ công với các mô hình Deep Learning hiện đại.

## 15. Recommended final topic statement

Final topic:

**A Comparative Study of Handcrafted Feature Extraction Methods for Vietnamese Traffic Sign Classification Using Traditional Machine Learning**

Vietnamese:

**So sánh các phương pháp trích xuất đặc trưng thủ công cho phân loại biển báo giao thông Việt Nam bằng học máy truyền thống**

Recommended main methods:

```text
Raw Pixel + SVM
Color Histogram + SVM
LBP + SVM
HOG + SVM
HOG + Color Histogram + SVM
```

Recommended main dataset:

```text
One Vietnamese traffic sign dataset with bounding box annotations
```

Recommended evaluation:

```text
Accuracy
Macro Precision
Macro Recall
Macro F1-score
Confusion matrix
Inference time
Error analysis
```

## 16. Final key message

The project is defensible because it is:

1. Focused.
2. Reproducible.
3. Aligned with the teacher's suggestion about feature extraction.
4. Compliant with the "no Deep Learning" requirement.
5. Based on Vietnamese traffic sign data.
6. Designed as a controlled comparative experiment.
7. Honest about its novelty and limitations.

Most important sentence:

> The goal is not to invent a new algorithm, but to evaluate which traditional handcrafted feature works best for Vietnamese traffic sign classification under a controlled Machine Learning setup.

Vietnamese version:

> Mục tiêu không phải là phát minh thuật toán mới, mà là đánh giá đặc trưng thủ công truyền thống nào phù hợp nhất cho phân loại biển báo giao thông Việt Nam trong một thiết kế thí nghiệm học máy có kiểm soát.
