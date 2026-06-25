# Codex Implementation Prompt: VNTS Feature × Classifier Matrix Experiment

## 0. Purpose

This file is intended to be read by Codex/Gemini CLI or another coding agent.

The goal is to implement a full experimental pipeline for a course-level Machine Learning research project:

**Vietnamese Traffic Sign Classification Using Handcrafted Feature Extraction and Traditional Machine Learning**

The project must avoid Deep Learning and focus on a controlled comparison of handcrafted features and traditional classifiers.

---

## 1. Main Problem / Research Question

### English

**Which combination of handcrafted feature extraction method and traditional Machine Learning classifier performs best for Vietnamese traffic sign classification on the VNTS dataset?**

### Vietnamese

**Tổ hợp giữa phương pháp trích xuất đặc trưng thủ công và mô hình học máy truyền thống nào đạt hiệu quả tốt nhất cho bài toán phân loại biển báo giao thông Việt Nam trên bộ dữ liệu VNTS?**

---

## 2. Main Experimental Design

The experiment should evaluate a full **feature × classifier matrix**.

### Features

Use the following feature extraction methods:

1. Raw Pixel
2. Color Histogram
3. LBP
4. HOG
5. HOG + Color Histogram

### Classifiers

Use the following traditional classifiers:

1. SVM
2. KNN
3. Random Forest
4. Logistic Regression

### Matrix

| Feature \ Classifier | SVM | KNN | Random Forest | Logistic Regression |
|---|---:|---:|---:|---:|
| Raw Pixel | x | x | x | x |
| Color Histogram | x | x | x | x |
| LBP | x | x | x | x |
| HOG | x | x | x | x |
| HOG + Color Histogram | x | x | x | x |

This gives:

```text
5 features × 4 classifiers = 20 experiments
```

### Baseline

Use:

```text
Raw Pixel + SVM
```

as the main baseline.

The final report should compare every method against this baseline.

---

## 3. Dataset

### Dataset name

VNTS / Vietnamese Traffic Signs dataset.

### Dataset directory

The dataset is located at:

```text
paper/dataset_NVTS
```

The coding agent must inspect this directory and infer the dataset structure.

Possible formats may include:

- image folders by class,
- annotation files,
- CSV labels,
- YOLO annotation format,
- COCO JSON,
- Pascal VOC XML.

The agent must first inspect the directory structure before implementing the parser.

### Class filtering rule

Only use classes with more than 20 images.

Rule:

```text
Keep class if number_of_images > 20
```

Do not use classes with 20 images or fewer.

The goal is to avoid very small classes that make training unstable.

### Dataset split

Use stratified splitting.

Recommended:

```text
Train: 80%
Test: 20%
```

If validation is needed:

```text
Train: 70%
Validation: 15%
Test: 15%
```

For this project, 80/20 train/test is acceptable.

Use:

```python
random_state = 42
```

for reproducibility.

---

## 4. Scope

### In scope

Implement:

1. Dataset inspection.
2. Class filtering by image count > 20.
3. Image loading.
4. Optional annotation parsing if needed.
5. Image preprocessing.
6. Feature extraction.
7. Feature normalization.
8. Traditional ML classifier training.
9. Evaluation.
10. Full feature × classifier matrix.
11. Result tables.
12. Confusion matrices.
13. Inference time measurement.
14. Final experiment summary.

### Out of scope

Do not implement:

1. CNN.
2. YOLO.
3. Faster R-CNN.
4. SSD.
5. RetinaNet.
6. Vision Transformer.
7. Any Deep Learning model.
8. Neural network classifier as the main method.
9. Full real-time traffic sign detection.

This project is strictly traditional Machine Learning.

---

## 5. Preprocessing Requirements

### 5.1 Resize

All images must be resized to:

```text
64 × 64
```

before feature extraction.

Use this size consistently for all experiments.

### 5.2 Color space and grayscale handling

Depending on the feature:

| Feature | Required image format |
|---|---|
| Raw Pixel | grayscale 64×64 |
| Color Histogram | HSV 64×64 |
| LBP | grayscale 64×64 |
| HOG | grayscale 64×64 or selected HOG preprocessing |
| HOG + Color Histogram | selected HOG preprocessing for HOG + HSV for Color Histogram |

### 5.3 Preprocessing variants: CLAHE, HUE, YUV

A previous paper compared preprocessing techniques such as:

- CLAHE
- HUE
- YUV

That paper found YUV-HOG to perform well, but this project should not assume that blindly.

Implement a preprocessing selection step for HOG-based features.

Recommended practical approach:

1. Run a small comparison using HOG + SVM with these preprocessing variants:
   - Default grayscale HOG
   - CLAHE + HOG
   - HUE/Hue-channel + HOG
   - YUV/Y-channel + HOG

2. Select the variant with the highest validation/test macro F1. If macro F1 ties, use accuracy as tie-breaker.

3. Use the best preprocessing variant for:
   - HOG
   - HOG + Color Histogram

If implementing this selection step is too time-consuming, use YUV/Y-channel before HOG as the default because the referenced HOG-SVM preprocessing paper reported strong results with YUV.

However, prefer implementing the actual comparison if possible.

### 5.4 Suggested definitions

#### Default grayscale

```text
image → resize 64×64 → grayscale
```

#### CLAHE

```text
image → resize 64×64 → grayscale → CLAHE
```

Suggested OpenCV parameters:

```python
clipLimit = 2.0
tileGridSize = (8, 8)
```

#### HUE

```text
image → resize 64×64 → HSV → use H channel
```

Alternative:

```text
image → HSV → equalize H channel → grayscale-like single channel feature input
```

Keep it simple and document the exact implementation.

#### YUV

```text
image → resize 64×64 → YUV → use Y channel
```

The Y channel represents luminance.

---

## 6. Feature Extraction Methods

Implement all feature extractors as separate modular functions.

Recommended API:

```python
def extract_raw_pixel(image) -> np.ndarray:
    ...

def extract_color_histogram(image) -> np.ndarray:
    ...

def extract_lbp(image) -> np.ndarray:
    ...

def extract_hog(image, preprocessing="best") -> np.ndarray:
    ...

def extract_hog_color(image, preprocessing="best") -> np.ndarray:
    ...
```

### 6.1 Raw Pixel

Pipeline:

```text
image → resize 64×64 → grayscale → flatten
```

Feature dimension:

```text
64 × 64 = 4096
```

This is the baseline feature.

### 6.2 Color Histogram

Pipeline:

```text
image → resize 64×64 → HSV → histogram per channel → concatenate
```

Suggested parameters:

```text
16 bins per channel
```

Feature dimension:

```text
16 × 3 = 48
```

Normalize histogram so that images with different brightness/intensity scale do not dominate.

### 6.3 LBP

Pipeline:

```text
image → resize 64×64 → grayscale → Local Binary Pattern → histogram
```

Suggested parameters:

```python
radius = 1
n_points = 8 * radius
method = "uniform"
```

Normalize the LBP histogram.

### 6.4 HOG

Pipeline:

```text
image → resize 64×64 → best HOG preprocessing → HOG descriptor
```

Suggested parameters:

```python
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
block_norm = "L2-Hys"
```

### 6.5 HOG + Color Histogram

Pipeline:

```text
HOG feature + Color Histogram feature → concatenate
```

Important:

- Extract HOG from grayscale/YUV/CLAHE/HUE depending on selected best preprocessing.
- Extract Color Histogram from HSV.
- Concatenate both vectors.
- Apply normalization after concatenation.

---

## 7. Feature Normalization

Normalize all feature vectors for all classifiers.

Even if some models such as Random Forest do not strictly need scaling, use a consistent pipeline for simplicity and fair processing.

Use:

```python
StandardScaler()
```

Each experiment should be implemented as:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", classifier)
])
```

This is especially important for:

- SVM
- KNN
- Logistic Regression

For Random Forest, scaling is not required but acceptable for consistency.

---

## 8. Classifiers

Use traditional ML classifiers only.

### 8.1 SVM

Preferred first option:

```python
LinearSVC(C=1.0, random_state=42, max_iter=10000)
```

Alternative if dataset is small enough:

```python
SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
```

Recommended:

Use `LinearSVC` first for speed. Optionally add RBF SVM later.

### 8.2 KNN

Use:

```python
KNeighborsClassifier(n_neighbors=5)
```

Optional tuning:

```text
k = 3, 5, 7
```

But start with 5.

### 8.3 Random Forest

Use:

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

### 8.4 Logistic Regression

Use:

```python
LogisticRegression(
    max_iter=5000,
    random_state=42,
    n_jobs=-1,
    multi_class="auto"
)
```

If convergence warnings occur, increase max_iter.

---

## 9. Metrics

Evaluate every experiment using:

1. Accuracy
2. Macro Precision
3. Macro Recall
4. Macro F1-score
5. Weighted F1-score
6. Confusion matrix
7. Training time
8. Inference time per image

### Main ranking metric

Use:

```text
Macro F1-score
```

as the main ranking metric.

Reason:

- Dataset may be imbalanced.
- Macro F1 treats all classes equally.

Accuracy should still be reported.

### Result table columns

The final result table should contain:

| feature | classifier | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | train_time_sec | inference_time_ms_per_image |
|---|---|---:|---:|---:|---:|---:|---:|---:|

Also add:

```text
rank_by_macro_f1
```

---

## 10. Output Requirements

The implementation should produce the following outputs.

### 10.1 Dataset summary

Save:

```text
paper/results/dataset_summary.csv
paper/results/class_distribution.png
paper/results/selected_classes.txt
```

The dataset summary should include:

- class name
- number of images
- whether selected or filtered out

### 10.2 Preprocessing comparison

If implemented, save:

```text
paper/results/preprocessing_comparison_hog_svm.csv
```

Columns:

| preprocessing | accuracy | macro_f1 |
|---|---:|---:|

Also save the selected preprocessing:

```text
paper/results/selected_hog_preprocessing.txt
```

### 10.3 Feature × Classifier matrix results

Save:

```text
paper/results/feature_classifier_matrix_results.csv
```

This is the main result file.

### 10.4 Pivot tables

Create pivot tables for easy paper writing.

Save:

```text
paper/results/pivot_accuracy.csv
paper/results/pivot_macro_f1.csv
```

Format:

| Feature \ Classifier | SVM | KNN | Random Forest | Logistic Regression |
|---|---:|---:|---:|---:|
| Raw Pixel | ... | ... | ... | ... |
| Color Histogram | ... | ... | ... | ... |
| LBP | ... | ... | ... | ... |
| HOG | ... | ... | ... | ... |
| HOG + Color Histogram | ... | ... | ... | ... |

### 10.5 Confusion matrices

Save confusion matrix images for all experiments or at least for:

1. Baseline: Raw Pixel + SVM
2. Best overall method
3. HOG + SVM
4. HOG + Color Histogram + SVM

Suggested directory:

```text
paper/results/confusion_matrices/
```

Filename format:

```text
confusion_matrix__{feature}__{classifier}.png
```

### 10.6 Classification reports

Save classification reports:

```text
paper/results/classification_reports/
```

Filename format:

```text
classification_report__{feature}__{classifier}.txt
```

### 10.7 Best model

Save the best model pipeline:

```text
paper/results/best_model.joblib
```

Also save metadata:

```text
paper/results/best_model_info.json
```

Include:

- feature
- classifier
- preprocessing
- accuracy
- macro_f1
- selected classes
- image size
- timestamp

---

## 11. Recommended Repository Structure

Assume the project has:

```text
paper/
  dataset_NVTS/
  results/
  src/
  notebooks/
```

Create if missing:

```text
paper/results/
paper/results/confusion_matrices/
paper/results/classification_reports/
paper/results/misclassified_samples/
paper/src/
```

Recommended code structure:

```text
paper/src/
  dataset.py
  preprocessing.py
  features.py
  models.py
  evaluate.py
  run_experiments.py
```

If the repository already has a different structure, adapt to it but keep the outputs under:

```text
paper/results/
```

---

## 12. Main Implementation Steps

### Step 1: Inspect dataset

Inspect:

```text
paper/dataset_NVTS
```

Determine:

- where images are stored,
- how labels are represented,
- whether images are already organized by class,
- whether bounding boxes exist,
- whether crops already exist.

Write findings to:

```text
paper/results/dataset_inspection.txt
```

### Step 2: Build metadata table

Create a metadata table:

```text
image_path,label,class_id
```

If bounding boxes are needed:

```text
image_path,label,class_id,xmin,ymin,xmax,ymax
```

Save:

```text
paper/results/dataset_metadata.csv
```

### Step 3: Filter classes

Keep only classes with more than 20 images.

Save:

```text
paper/results/filtered_metadata.csv
```

### Step 4: Train/test split

Use stratified split.

Save:

```text
paper/results/train_metadata.csv
paper/results/test_metadata.csv
```

### Step 5: Optional preprocessing selection for HOG

Run HOG + SVM for:

1. default grayscale
2. CLAHE
3. HUE
4. YUV

Rank by macro F1.

Save:

```text
paper/results/preprocessing_comparison_hog_svm.csv
paper/results/selected_hog_preprocessing.txt
```

### Step 6: Run all 20 experiments

For each pair:

```text
feature × classifier
```

do:

1. Extract train features.
2. Extract test features.
3. Fit scaler + classifier pipeline.
4. Predict.
5. Compute metrics.
6. Save report.
7. Save confusion matrix.
8. Record training time.
9. Record inference time.

### Step 7: Summarize results

Create:

```text
paper/results/feature_classifier_matrix_results.csv
paper/results/pivot_accuracy.csv
paper/results/pivot_macro_f1.csv
```

### Step 8: Save best model

Use macro F1 to select best method.

Save:

```text
paper/results/best_model.joblib
paper/results/best_model_info.json
```

---

## 13. Suggested Command

Create a runnable script:

```bash
python paper/src/run_experiments.py
```

Optional arguments:

```bash
python paper/src/run_experiments.py \
  --dataset paper/dataset_NVTS \
  --output paper/results \
  --image-size 64 \
  --min-images-per-class 21 \
  --random-state 42
```

Note:

Because the rule is "classes with more than 20 images", the minimum count is:

```text
21
```

---

## 14. Expected Interpretation

After results are generated, the paper should discuss:

1. Which feature performs best on average.
2. Which classifier performs best on average.
3. Which feature-classifier pair performs best overall.
4. Whether HOG + SVM remains a strong baseline.
5. Whether HOG + Color improves over HOG alone.
6. Whether Color Histogram alone is too weak because many signs share the same colors.
7. Whether LBP is weaker because traffic signs are more shape-based than texture-based.
8. Which classes are most often confused.
9. Whether preprocessing such as YUV or CLAHE improves HOG performance.

---

## 15. Important Scientific Framing

Do not claim:

```text
We propose a completely new algorithm.
```

Instead claim:

```text
We perform a controlled empirical comparison of handcrafted features and traditional classifiers for Vietnamese traffic sign classification.
```

The strongest contribution is:

```text
A full feature × classifier matrix on VNTS under a no-Deep-Learning constraint.
```

---

## 16. Paper-ready Result Discussion Template

Use this structure later in the paper:

1. **Baseline comparison**
   - Compare all methods against Raw Pixel + SVM.

2. **Feature analysis**
   - Discuss whether HOG, LBP, Color Histogram, or HOG + Color works best.

3. **Classifier analysis**
   - Discuss which classifier performs best across features.

4. **Best method**
   - Identify the highest macro F1 method.

5. **Confusion matrix**
   - Explain common mistakes.

6. **Efficiency**
   - Compare inference time.

7. **Limitations**
   - Only classification is evaluated.
   - Detection from full dashcam images is not the main focus.
   - Dataset may be imbalanced.
   - Results are limited to selected classes with more than 20 images.

---

## 17. Minimum Success Criteria

The task is successful if the code produces:

1. A filtered VNTS metadata file.
2. A class distribution summary.
3. A train/test split.
4. Feature extraction for all 5 features.
5. Training/evaluation for all 4 classifiers.
6. A 20-row result CSV.
7. Accuracy and macro F1 pivot tables.
8. Confusion matrix for baseline and best model.
9. Best model info JSON.
10. A short text summary of the best method.

---

## 18. Final Instruction for Codex

Please implement the project according to this document.

Start by inspecting the dataset structure under:

```text
paper/dataset_NVTS
```

Then create the necessary scripts under:

```text
paper/src/
```

and save all outputs under:

```text
paper/results/
```

Keep the implementation simple, reproducible, and strictly based on traditional Machine Learning. Do not use Deep Learning.
