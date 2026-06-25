# Project Implementation Plan: Vietnamese Traffic Sign Classification Using Handcrafted Features and Traditional Machine Learning

## 0. Purpose of this document

This file is a project guide for CLI coding agents such as Codex, Gemini CLI, or other development assistants.

The goal is to help agents understand the research direction, technical scope, implementation stages, expected code structure, experiments, and final paper requirements.

## 1. Research direction

### Working title

**A Comparative Study of Handcrafted Feature Extraction Methods for Vietnamese Traffic Sign Classification Using Traditional Machine Learning**

Vietnamese title:

**So sánh các phương pháp trích xuất đặc trưng thủ công cho phân loại biển báo giao thông Việt Nam bằng học máy truyền thống**

### Main research idea

This project studies traffic sign classification using **traditional Machine Learning only**, without Deep Learning.

The main focus is not to build a complete end-to-end autonomous driving system. Instead, the project focuses on the **recognition/classification stage** of traffic sign recognition.

The dataset already contains images and bounding box annotations. We use the bounding boxes to crop traffic signs, then compare different handcrafted feature extraction methods with traditional classifiers.

### Main research question

**Which handcrafted feature extraction method is most effective for Vietnamese traffic sign classification using traditional Machine Learning?**

### Vietnamese version

**Phương pháp trích xuất đặc trưng thủ công nào hiệu quả nhất cho bài toán phân loại biển báo giao thông Việt Nam bằng học máy truyền thống?**

### Expected contribution

This project does not claim to propose a completely new traffic sign recognition algorithm.

The contribution is an **empirical comparison** of handcrafted feature extraction methods for Vietnamese traffic sign classification.

Possible contribution statement:

> This study provides an empirical comparison of handcrafted feature extraction methods for Vietnamese traffic sign classification using traditional Machine Learning classifiers.

Vietnamese version:

> Nghiên cứu này thực hiện so sánh thực nghiệm các phương pháp trích xuất đặc trưng thủ công cho bài toán phân loại biển báo giao thông Việt Nam bằng các mô hình học máy truyền thống.

## 2. Scope

### In scope

The project should focus on:

1. Vietnamese traffic sign classification.
2. Cropping traffic signs using existing bounding box annotations.
3. Handcrafted feature extraction.
4. Traditional Machine Learning classifiers.
5. Experimental comparison between feature extraction methods.
6. Accuracy, macro F1-score, precision, recall, confusion matrix, and inference time.
7. Writing a course-level research paper.

### Out of scope

Do not focus on the following as the main research contribution:

1. Full object detection from raw dashcam images.
2. YOLO, CNN, Faster R-CNN, RetinaNet, SSD, or any Deep Learning model.
3. Real-time embedded deployment.
4. Training a neural network.
5. Creating a new large-scale dataset.
6. Building a production ADAS system.

### Important constraint

**Do not use Deep Learning.**

Avoid:

- CNN
- YOLO
- Faster R-CNN
- SSD
- RetinaNet
- Vision Transformer
- MLP as the main classifier, unless only mentioned in related work

Recommended models:

- SVM
- LinearSVC
- RBF SVM
- KNN
- Random Forest
- Logistic Regression
- Decision Tree

## 3. Recommended project framing

The safest framing for the paper is:

> In a complete traffic sign recognition system, both detection and classification are required. However, this study focuses on the classification stage. Traffic sign regions are cropped using available bounding box annotations. The main objective is to compare handcrafted feature extraction methods for Vietnamese traffic sign classification using traditional Machine Learning classifiers.

Vietnamese version:

> Trong một hệ thống nhận dạng biển báo giao thông hoàn chỉnh, cả phát hiện vị trí và phân loại biển báo đều cần thiết. Tuy nhiên, nghiên cứu này tập trung vào giai đoạn phân loại. Vùng biển báo được crop bằng bounding box annotation có sẵn. Mục tiêu chính là so sánh các phương pháp trích xuất đặc trưng thủ công cho phân loại biển báo giao thông Việt Nam bằng các mô hình học máy truyền thống.

## 4. Related work summary

The project direction is inspired by several related papers.

### 4.1 Can Tho University 2015 paper

Paper:

**Road traffic sign detection and recognition using HOG feature and Artificial Neural Network**

Key points:

- Vietnamese traffic sign context.
- Uses image processing and Machine Learning.
- Detects candidate regions using color segmentation, edge detection, and shape analysis.
- Extracts HOG features.
- Classifies using MLP and SVM.
- Reports around 94% recognition accuracy.
- Processes frames from 31 videos.
- Very relevant to Vietnamese traffic signs and HOG/SVM.

How our project differs:

- We do not focus on full detection from video.
- We use existing bounding boxes for cropping.
- We focus more explicitly on comparing multiple handcrafted feature extraction methods.
- We avoid MLP/neural network as the main approach.

### 4.2 HOG-SVM preprocessing comparison paper

Paper:

**Comparing Performance of Preprocessing Techniques for Traffic Sign Recognition Using a HOG-SVM**

Key points:

- Uses GTSRB dataset.
- Uses HOG + SVM.
- Compares preprocessing techniques such as CLAHE, HUE, and YUV.
- Reports that YUV-HOG improves accuracy from about 89.65% to 91.25%.
- Very relevant to the idea of improving feature extraction/preprocessing in traditional Machine Learning.

How our project differs:

- We use Vietnamese traffic sign data instead of GTSRB.
- We compare handcrafted features such as Raw Pixel, Color Histogram, LBP, HOG, and HOG + Color.
- We focus on Vietnamese traffic sign classification.

### 4.3 SHS 2022 classifier comparison paper

Paper:

**Research on the Optimal Machine Learning Classifier for Traffic Signs**

Key points:

- Compares SVM, MLP, and Logistic Regression.
- Studies preprocessing such as binarization and Laplacian sharpening.
- Finds SVM generally performs strongly.
- Useful as related work for traditional classifier comparison.

How our project differs:

- We focus more on feature extraction comparison.
- We avoid MLP as a main method.
- We use Vietnamese traffic sign data if possible.

### 4.4 FAIR 2025 YOLOv10 Vietnam paper

Paper:

**Traffic Sign Detection in Vietnam: An Experimental Study**

Key points:

- Uses YOLOv10.
- Vietnamese traffic sign detection.
- Large dataset of more than 57,000 images.
- Uses ROI cropping, sliding window, augmentation.
- Reports mAP@0.5 around 88.2% and about 48 FPS on RTX 3090.

How our project uses it:

- Cite this paper for motivation and Vietnamese traffic sign context.
- Mention that Deep Learning performs well but requires more compute and is outside the scope of this course project.
- Do not follow its YOLO methodology.

## 5. Dataset plan

### Recommended dataset strategy

Use **one main Vietnamese traffic sign dataset** with bounding box annotations.

Using one dataset is enough for this course-level paper. Multiple datasets are optional and may make the project unnecessarily complex.

### Dataset requirements

The dataset should have:

1. Images of Vietnamese traffic signs.
2. Bounding box annotations.
3. Class labels.
4. Enough examples per selected class.
5. A format that can be parsed: YOLO, COCO, Pascal VOC, or CSV.

### Suggested class selection

If the dataset has many classes, do not necessarily use all classes.

Recommended options:

#### Option A: Simple and stable

Use the **10 most frequent classes**.

#### Option B: Medium difficulty

Use all classes with at least **50 samples**.

#### Option C: Harder

Use all available classes.

Recommended choice:

**Use 10 to 15 classes with enough samples.**

Reason:

- Easier training.
- Less class imbalance.
- Confusion matrix remains readable.
- Easier analysis in the paper.

### Class imbalance handling

Use stratified train/test split.

Possible split:

- Train: 80%
- Test: 20%

Or:

- Train: 70%
- Validation: 15%
- Test: 15%

For this project, 80/20 is acceptable if time is limited.

## 6. Overall pipeline

The main pipeline should be:

```text
Dataset images + annotations
→ parse annotation files
→ crop traffic sign regions using bounding boxes
→ resize cropped signs to fixed size
→ preprocess images
→ extract handcrafted features
→ train traditional ML classifier
→ evaluate
→ compare methods
→ write paper
```

### Recommended input/output

Input:

```text
raw image + bbox annotation
```

Intermediate:

```text
cropped traffic sign image
```

Model input:

```text
feature vector extracted from cropped image
```

Output:

```text
predicted traffic sign class
```

## 7. Data preprocessing

### 7.1 Annotation parsing

Support possible annotation formats:

#### YOLO format

```text
class_id x_center y_center width height
```

Usually normalized by image width and height.

Need convert to pixel coordinates:

```text
x_min = (x_center - width / 2) * image_width
y_min = (y_center - height / 2) * image_height
x_max = (x_center + width / 2) * image_width
y_max = (y_center + height / 2) * image_height
```

#### COCO format

```text
bbox = [x, y, width, height]
```

#### Pascal VOC format

```xml
<xmin>...</xmin>
<ymin>...</ymin>
<xmax>...</xmax>
<ymax>...</ymax>
```

### 7.2 Cropping

For each object annotation:

```text
image + bounding box → crop traffic sign
```

Save each cropped sign as a separate image.

Recommended output folder:

```text
data/
  crops/
    class_001/
      img_000001.jpg
      img_000002.jpg
    class_002/
      img_000003.jpg
```

Alternative: save all crop paths and labels to CSV.

Recommended CSV:

```text
crop_path,label,class_id,source_image,xmin,ymin,xmax,ymax
```

### 7.3 Resize

Resize every cropped image to a fixed size.

Recommended:

```text
64x64
```

Alternative:

```text
128x128
```

Start with 64x64 because it is faster for HOG and SVM.

### 7.4 Color conversion

Depending on feature extraction method:

- Raw Pixel: grayscale or RGB
- Color Histogram: HSV
- LBP: grayscale
- HOG: grayscale
- HOG + Color: grayscale for HOG, HSV for color histogram

### 7.5 Normalization

Normalize feature vectors before training.

Use:

```python
StandardScaler()
```

For SVM, scaling is important.

## 8. Feature extraction methods

Keep the classifier fixed at first, preferably SVM, so the experiment focuses on feature extraction.

### Method 1: Raw Pixel

Purpose: simple baseline.

Pipeline:

```text
crop → resize 64x64 → grayscale → flatten → StandardScaler → SVM
```

Feature dimension:

```text
64 × 64 = 4096
```

Expected behavior:

- Simple.
- Fast to implement.
- Sensitive to lighting, rotation, and crop quality.
- Useful as baseline.

### Method 2: Color Histogram

Purpose: use color distribution.

Pipeline:

```text
crop → resize 64x64 → convert to HSV → histogram per channel → concatenate → StandardScaler → SVM
```

Suggested parameters:

```text
H bins: 16 or 32
S bins: 16 or 32
V bins: 16 or 32
```

Simpler version:

```text
16 bins per channel → 48-dimensional feature
```

Expected behavior:

- Useful because traffic signs have strong color patterns.
- Weak at distinguishing signs with same color but different internal symbols.
- May work well for broad categories: prohibition, warning, mandatory, guide.

### Method 3: LBP

Purpose: local texture/pattern feature.

Pipeline:

```text
crop → resize 64x64 → grayscale → LBP → LBP histogram → StandardScaler → SVM
```

Suggested parameters:

```text
radius = 1 or 2
n_points = 8 * radius
method = "uniform"
```

Expected behavior:

- Lightweight.
- Captures local texture.
- May not be as strong as HOG for shape-heavy objects.

### Method 4: HOG

Purpose: main method.

Pipeline:

```text
crop → resize 64x64 → grayscale → HOG → StandardScaler → SVM
```

Suggested HOG parameters:

```text
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
block_norm = "L2-Hys"
```

Expected behavior:

- Strong traditional feature for traffic signs.
- Captures edges, gradients, and shape.
- Likely to outperform Raw Pixel, Color Histogram, and LBP.

### Method 5: HOG + Color Histogram

Purpose: combine shape and color.

Pipeline:

```text
HOG feature + HSV color histogram → concatenate → StandardScaler → SVM
```

Expected behavior:

- Often better than HOG alone if color is useful.
- Good final method for comparison.

### Optional Method 6: HOG + LBP

Pipeline:

```text
HOG feature + LBP histogram → concatenate → StandardScaler → SVM
```

Use only if time allows.

## 9. Classifier plan

### Main classifier

Use SVM.

Start with:

```python
LinearSVC()
```

Then optionally test:

```python
SVC(kernel="rbf")
```

### Why SVM?

- Traditional Machine Learning.
- Common in traffic sign recognition.
- Works well with high-dimensional handcrafted features such as HOG.
- Easier to justify in the paper.

### Optional classifier comparison

If the project has time, after finding the best feature, compare classifiers:

```text
Best feature + KNN
Best feature + Random Forest
Best feature + Logistic Regression
Best feature + SVM
```

This should be secondary. The main experiment should remain feature comparison.

## 10. Experiment design

### Main experiment: feature comparison with fixed classifier

Use SVM for all feature methods.

| Experiment | Feature | Classifier | Purpose |
|---|---|---|---|
| E1 | Raw Pixel | SVM | Baseline |
| E2 | Color Histogram | SVM | Color-only feature |
| E3 | LBP | SVM | Texture feature |
| E4 | HOG | SVM | Shape/gradient feature |
| E5 | HOG + Color Histogram | SVM | Shape + color |
| E6 optional | HOG + LBP | SVM | Shape + texture |

### Secondary experiment: classifier comparison

Only if time allows.

| Experiment | Feature | Classifier |
|---|---|---|
| C1 | Best feature | KNN |
| C2 | Best feature | Random Forest |
| C3 | Best feature | Logistic Regression |
| C4 | Best feature | SVM |

## 11. Evaluation metrics

Report at least:

1. Accuracy
2. Macro Precision
3. Macro Recall
4. Macro F1-score
5. Confusion matrix
6. Training time
7. Inference time per image

### Why macro F1?

If the dataset is imbalanced, accuracy alone can be misleading.

Macro F1 treats each class equally, so it is better for class imbalance.

### Suggested result table

| Method | Feature | Classifier | Accuracy | Macro Precision | Macro Recall | Macro F1 | Inference time/image |
|---|---|---|---:|---:|---:|---:|---:|
| Baseline | Raw Pixel | SVM | TBD | TBD | TBD | TBD | TBD |
| Method 1 | Color Histogram | SVM | TBD | TBD | TBD | TBD | TBD |
| Method 2 | LBP | SVM | TBD | TBD | TBD | TBD | TBD |
| Method 3 | HOG | SVM | TBD | TBD | TBD | TBD | TBD |
| Method 4 | HOG + Color | SVM | TBD | TBD | TBD | TBD | TBD |

## 12. Error analysis

After training models, inspect misclassified examples.

Look for:

1. Similar signs with different numbers.
2. Similar shape and color.
3. Low resolution crops.
4. Blurred images.
5. Bad lighting.
6. Occluded signs.
7. Incorrect or loose bounding boxes.
8. Very small traffic signs.
9. Imbalanced classes.

Common examples:

- Speed limit signs confused with other speed limits.
- No left turn confused with no U-turn.
- Warning signs confused with other triangular signs.
- Mandatory blue signs confused with other blue signs.

The paper should include a short discussion of these errors.

## 13. Recommended repository structure

Use this structure:

```text
project-root/
  README.md
  requirements.txt
  configs/
    default.yaml

  data/
    raw/
    annotations/
    crops/
    processed/
    splits/
      train.csv
      val.csv
      test.csv

  notebooks/
    01_dataset_exploration.ipynb
    02_feature_extraction_debug.ipynb
    03_model_experiments.ipynb

  src/
    __init__.py

    data/
      parse_annotations.py
      crop_signs.py
      make_splits.py
      dataset_summary.py

    features/
      raw_pixels.py
      color_histogram.py
      lbp.py
      hog.py
      combined.py

    models/
      train.py
      evaluate.py
      predict.py

    utils/
      image_io.py
      metrics.py
      plotting.py
      timing.py

  experiments/
    results/
      metrics.csv
      confusion_matrices/
      misclassified_samples/

  paper/
    figures/
    tables/
    draft.md

  reports/
    dataset_summary.md
    experiment_summary.md
```

## 14. Suggested Python packages

Use:

```text
numpy
pandas
opencv-python
scikit-image
scikit-learn
matplotlib
tqdm
joblib
PyYAML
```

Optional:

```text
seaborn
```

However, if plotting inside ChatGPT tools, do not use seaborn unless explicitly allowed. For local development, seaborn is acceptable if the team wants it.

## 15. Implementation tasks for CLI agents

### Task 1: Dataset exploration

Create a script:

```text
src/data/dataset_summary.py
```

It should:

1. Count number of images.
2. Count number of objects per class.
3. Count number of images per class.
4. Show top 10 or top 15 most frequent classes.
5. Save a class distribution CSV.
6. Save a class distribution chart.

Output:

```text
reports/dataset_summary.md
experiments/results/class_distribution.csv
paper/figures/class_distribution.png
```

### Task 2: Annotation parser

Create:

```text
src/data/parse_annotations.py
```

It should support the chosen dataset format.

Output should be a standard CSV:

```text
image_path,class_id,label,xmin,ymin,xmax,ymax
```

### Task 3: Crop signs

Create:

```text
src/data/crop_signs.py
```

It should:

1. Read standard annotation CSV.
2. Crop each traffic sign.
3. Clip bbox coordinates to image boundaries.
4. Resize optionally.
5. Save crops.
6. Save metadata CSV.

Output CSV:

```text
data/processed/crops_metadata.csv
```

### Task 4: Train/test split

Create:

```text
src/data/make_splits.py
```

It should:

1. Read crops metadata.
2. Filter selected classes.
3. Use stratified split.
4. Save train/test CSV files.

Output:

```text
data/splits/train.csv
data/splits/test.csv
```

Optional:

```text
data/splits/val.csv
```

### Task 5: Feature extraction modules

Create separate feature extractors:

```text
src/features/raw_pixels.py
src/features/color_histogram.py
src/features/lbp.py
src/features/hog.py
src/features/combined.py
```

Each should expose a consistent function:

```python
def extract_features(image) -> np.ndarray:
    ...
```

Or batch version:

```python
def extract_features_from_paths(image_paths: list[str]) -> np.ndarray:
    ...
```

### Task 6: Training script

Create:

```text
src/models/train.py
```

It should:

1. Load train/test CSV.
2. Extract selected feature type.
3. Scale features with StandardScaler.
4. Train classifier.
5. Save trained model pipeline.
6. Save metrics.

Suggested command:

```bash
python -m src.models.train --feature hog --classifier linear_svm
```

Output:

```text
experiments/results/hog_linear_svm_model.joblib
experiments/results/metrics.csv
```

### Task 7: Evaluation script

Create:

```text
src/models/evaluate.py
```

It should:

1. Load saved model.
2. Predict test set.
3. Compute metrics.
4. Save classification report.
5. Save confusion matrix.
6. Save misclassified samples.

Output:

```text
experiments/results/classification_report_hog_svm.txt
experiments/results/confusion_matrices/hog_svm.png
experiments/results/misclassified_samples/hog_svm/
```

### Task 8: Experiment runner

Create:

```text
src/models/run_all_experiments.py
```

It should run:

1. Raw Pixel + SVM
2. Color Histogram + SVM
3. LBP + SVM
4. HOG + SVM
5. HOG + Color Histogram + SVM
6. Optional HOG + LBP + SVM

Output:

```text
experiments/results/metrics_summary.csv
```

## 16. Coding guidelines

### General

1. Keep code modular.
2. Avoid hard-coded paths.
3. Use a config file.
4. Save all results.
5. Make experiments reproducible.
6. Set random seed.

### Recommended random seed

```python
RANDOM_STATE = 42
```

### Use scikit-learn Pipeline

Recommended:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LinearSVC(random_state=42))
])
```

### Save models

Use:

```python
joblib.dump(model, "path/to/model.joblib")
```

### Metrics

Use:

```python
classification_report
accuracy_score
precision_recall_fscore_support
confusion_matrix
```

## 17. Recommended configuration file

Create:

```text
configs/default.yaml
```

Example:

```yaml
project:
  random_state: 42

data:
  raw_dir: "data/raw"
  annotation_dir: "data/annotations"
  crops_dir: "data/crops"
  processed_dir: "data/processed"
  splits_dir: "data/splits"
  image_size: [64, 64]
  min_samples_per_class: 50
  max_classes: 15
  test_size: 0.2

features:
  hog:
    orientations: 9
    pixels_per_cell: [8, 8]
    cells_per_block: [2, 2]
    block_norm: "L2-Hys"

  color_histogram:
    color_space: "HSV"
    bins_per_channel: 16

  lbp:
    radius: 1
    n_points: 8
    method: "uniform"

model:
  classifier: "linear_svm"
  svm:
    C: 1.0
    max_iter: 5000
```

## 18. Minimum viable project

If time is limited, implement only this:

1. Parse annotations.
2. Crop signs using bounding boxes.
3. Select 10 most frequent classes.
4. Split train/test.
5. Implement Raw Pixel + SVM.
6. Implement HOG + SVM.
7. Implement HOG + Color Histogram + SVM.
8. Report accuracy, macro F1, and confusion matrix.
9. Write paper.

Minimum experiment table:

| Method | Feature | Classifier |
|---|---|---|
| Baseline | Raw Pixel | SVM |
| Main 1 | HOG | SVM |
| Main 2 | HOG + Color Histogram | SVM |

## 19. Full project plan

If time allows, implement:

1. Raw Pixel + SVM
2. Color Histogram + SVM
3. LBP + SVM
4. HOG + SVM
5. HOG + Color Histogram + SVM
6. HOG + LBP + SVM
7. Best feature + KNN
8. Best feature + Random Forest
9. Best feature + Logistic Regression
10. Best feature + SVM

## 20. Paper outline

### 1. Abstract

Mention:

- Vietnamese traffic sign classification.
- Traditional Machine Learning.
- Handcrafted feature extraction.
- SVM classifier.
- Main result.

### 2. Introduction

Include:

- Traffic sign recognition is important for driver assistance.
- Deep Learning is powerful but may require high computational resources.
- Traditional Machine Learning remains useful for constrained settings and course scope.
- This study focuses on classification from cropped traffic signs.
- Main objective: compare handcrafted features.

### 3. Related Work

Discuss:

1. Traditional TSR methods: HOG, SVM, LBP, color features.
2. Vietnamese traffic sign research with HOG + SVM/MLP.
3. HOG-SVM preprocessing studies.
4. Deep Learning methods such as YOLO only as contrast, not as project method.

### 4. Dataset

Include:

- Dataset name.
- Number of images.
- Number of classes.
- Annotation format.
- Selected classes.
- Train/test split.
- Class distribution chart.
- Example cropped images.

### 5. Methodology

Subsections:

1. Data preprocessing.
2. Cropping using bounding boxes.
3. Feature extraction:
   - Raw Pixel
   - Color Histogram
   - LBP
   - HOG
   - HOG + Color
4. Classifier:
   - SVM
5. Evaluation metrics.

### 6. Experiments and Results

Include:

- Main result table.
- Confusion matrix.
- Training/inference time.
- Best method.

### 7. Discussion

Discuss:

- Which feature worked best.
- Why HOG or HOG + Color may work well.
- Which classes are confused.
- Effect of class imbalance.
- Limitations.

### 8. Conclusion and Future Work

Mention:

- Summary of best feature.
- Traditional ML can classify traffic signs reasonably.
- Limitations: not full detection, not deep learning, limited dataset.
- Future work: full detection, larger dataset, robust preprocessing, real-time demo, cross-dataset evaluation.

## 21. Suggested paper wording

### Scope statement

> This study focuses on the recognition stage of traffic sign recognition. Instead of performing full object detection from raw dashcam images, traffic sign regions are cropped using the available bounding box annotations. This allows the study to focus on the effect of handcrafted feature extraction methods on classification performance.

### Contribution statement

> The main contribution of this work is an empirical comparison of handcrafted feature extraction methods for Vietnamese traffic sign classification using traditional Machine Learning.

### Limitation statement

> Since this work uses ground-truth bounding boxes for cropping, it does not evaluate the full detection performance of a traffic sign recognition system. The results therefore reflect classification performance under the assumption that sign regions are already localized.

### Deep Learning contrast

> Although Deep Learning methods such as YOLO-based detectors have achieved strong performance in traffic sign detection, they require more computational resources and are outside the scope of this study. This work instead investigates lightweight traditional Machine Learning methods based on handcrafted features.

## 22. Suggested presentation structure

Slides:

1. Title
2. Problem and motivation
3. Research gap and scope
4. Dataset
5. Pipeline overview
6. Feature extraction methods
7. Classifier and evaluation metrics
8. Experiment setup
9. Results table
10. Confusion matrix
11. Error analysis
12. Conclusion and future work

## 23. Risk management

### Risk 1: Dataset format is difficult

Solution:

- Convert annotations to one standard CSV first.
- Do not write model code before dataset conversion is stable.

### Risk 2: Too many classes and low accuracy

Solution:

- Filter to top 10 or 15 classes.
- Use classes with enough samples.
- Report class imbalance honestly.

### Risk 3: HOG + SVM training is slow

Solution:

- Use 64x64 image size.
- Start with LinearSVC.
- Reduce number of classes.
- Cache extracted features to `.npy` files.

### Risk 4: Results are not high

Solution:

- Focus on comparative analysis, not absolute state-of-the-art.
- Explain limitations.
- Compare relative performance between features.
- Use macro F1 and confusion matrix.

### Risk 5: Teacher expects feature extraction focus

Solution:

- Keep classifier fixed.
- Make the main experiment feature comparison.
- Do not spend too much paper space on classifier tuning.

## 24. Final recommended approach

The recommended final approach is:

```text
Use one Vietnamese traffic sign dataset
→ crop signs using bounding box annotations
→ select 10 to 15 frequent classes
→ resize crops to 64x64
→ compare handcrafted features:
   1. Raw Pixel
   2. Color Histogram
   3. LBP
   4. HOG
   5. HOG + Color Histogram
→ use SVM as fixed classifier
→ evaluate with accuracy, macro F1, confusion matrix, inference time
→ write comparative study paper
```

## 25. Agent instructions

When an AI coding agent reads this file, it should:

1. Prioritize building a clean dataset preprocessing pipeline.
2. Avoid Deep Learning.
3. Implement feature extraction methods modularly.
4. Keep experiments reproducible.
5. Save all metrics and plots.
6. Generate outputs that can be directly used in the final paper.
7. Ask before changing the research scope.
8. Prefer simple, working experiments over overly complex methods.
9. Keep the main research focus on handcrafted feature extraction comparison.
10. Do not implement YOLO/CNN unless explicitly requested as a separate non-main baseline.

## 26. Expected final deliverables

The project should produce:

1. Clean cropped traffic sign dataset.
2. Dataset summary report.
3. Feature extraction code.
4. Training and evaluation scripts.
5. Result table.
6. Confusion matrices.
7. Error analysis samples.
8. Final paper draft.
9. Presentation slides.
10. Optional demo notebook.

## 27. Final note

This project is intended as a course-level research paper, not a conference-level novel contribution. The implementation should be practical, clear, reproducible, and easy to explain.

The strongest version of the project is not the most complex one. The strongest version is the one that clearly answers:

> Among several handcrafted feature extraction methods, which one works best for Vietnamese traffic sign classification using traditional Machine Learning?
