# Breast Cancer Classification Using CNN on Treemap-Encoded Gene Expression Data

This repository contains the complete implementation and documentation of a minor project focused on binary classification of breast cancer samples (tumor vs normal) using RNA-Seq gene expression data.  
The project explores and compares a traditional machine learning approach (Random Forest) with a deep learning approach (Convolutional Neural Network) by transforming gene expression profiles into treemap-encoded images.

---

## 1. Project Motivation

Breast cancer is one of the most prevalent cancers worldwide and is driven by abnormal changes in gene expression. RNA-Seq technology enables the measurement of expression levels for thousands of genes simultaneously, generating highly informative but **high-dimensional data**.

Traditional machine learning methods often require careful feature selection and may struggle to capture complex patterns. Deep learning models, particularly CNNs, excel at learning hierarchical patterns but are typically designed for image data.

This project bridges this gap by:
- Transforming gene expression data into image-like representations
- Applying CNNs to learn expression patterns
- Comparing performance with a strong machine learning baseline

---

## 2. Dataset Description

- **Data Source:** The Cancer Genome Atlas (TCGA)
- **Cancer Type:** Breast Invasive Carcinoma (BRCA)
- **Access Platform:** UCSC Xena Browser
- **Data Type:** Bulk RNA-Seq gene expression
- **File Format:** Tab-separated values (`.tsv`)

### Sample Composition
| Class | Number of Samples |
|-----|------------------|
| Tumor | 1,097 |
| Normal | 114 |
| **Total** | **1,211** |

The dataset is **imbalanced**, reflecting real-world cancer data distributions.

---

## 3. Data Preprocessing

The following preprocessing steps were applied:

1. **Log₂ normalization**  
   - Reduces skewness in expression values  
   - Stabilizes variance across genes  

2. **Variance-based gene selection**  
   - Genes with low variance were removed  
   - Selected:
     - Top **100 genes** for Random Forest
     - Top **5,000 genes** for CNN

3. **Train–Test Split**
   - 80% Training data
   - 20% Testing data

---

## 4. Treemap Encoding of Gene Expression

To make gene expression data suitable for CNN input:

- A **treemap layout** was created using 5,000 highly variable genes
- Each gene is assigned a **fixed rectangular region** in a **120 × 120 grid**
- The layout remains **identical for all samples**
- Gene expression values are mapped to **grayscale intensities**
  - Higher expression → brighter regions
  - Lower expression → darker regions

Each treemap image represents **one patient sample**.

---

## 5. Models Implemented

### 5.1 Random Forest (Baseline Model)

- Input: Top 100 most variable genes
- Handles non-linear relationships
- Robust to noise and high-dimensional data
- Class-weight balancing applied to handle data imbalance

**Purpose:**  
To serve as a strong traditional machine learning baseline.

---

### 5.2 Convolutional Neural Network (CNN)

- Input: 120 × 120 treemap-encoded grayscale images
- Architecture:
  - Convolutional layers with ReLU activation
  - MaxPooling layers
  - Dense layers for classification
  - Dropout for regularization
  - Sigmoid output layer
- Optimizer: Adam
- Loss Function: Binary Cross-Entropy
- Training:
  - 20 epochs
  - Class balancing applied

**Purpose:**  
To learn spatial and intensity-based patterns in gene expression data.

---

## 6. Evaluation Metrics

Models were evaluated on the independent test set using:

- Accuracy
- AUROC (Area Under ROC Curve)
- Confusion Matrix
- ROC Curves
- Training and Validation Curves (CNN)

AUROC was emphasized due to class imbalance.

---

## 7. Results

| Model | Input Representation | Accuracy | AUROC |
|-----|---------------------|----------|-------|
| Random Forest | Top 100 genes | 98.77% | 0.9996 |
| CNN | Treemap images (5,000 genes) | 99.59% | 0.9990 |

### Observations
- Both models achieved very high performance
- CNN showed slightly better accuracy
- Confusion matrices indicate very low false positives and false negatives
- CNN benefits from learning spatial expression patterns

---

## 8. Key Learnings

- Gene expression data requires careful preprocessing
- Feature representation significantly impacts model performance
- CNNs can effectively learn from non-traditional image data
- Traditional machine learning models remain strong baselines
- Evaluation metrics beyond accuracy are crucial in medical data

---

## 9. Limitations

- Treemap layout is spatially artificial
- CNN models are less interpretable
- Biological meaning of learned features is indirect
- Subtype classification not explored

---

## 10. Future Work

- Incorporate biological knowledge into treemap layouts (pathways, networks)
- Extend to breast cancer subtype classification
- Apply model interpretability techniques
- Evaluate on other cancer types or multi-omics datasets

---

## 11. Tools and Libraries

- Python
- NumPy, Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Squarify

---

## 12. References

- The Cancer Genome Atlas (TCGA)
- UCSC Xena Browser
- Breiman, L. (2001). Random Forests.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep Learning.

---

## 13. Author

**Khandakar Jianur Islam**  
M.Sc. Bioinformatics  
Jamia Millia Islamia University  

---

## 14. License

This project is intended for academic and educational purposes only.


