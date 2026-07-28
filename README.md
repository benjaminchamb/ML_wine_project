# Machine Learning 02450 – Wine Dataset Analysis

Coursework for 02450 Introduction to Machine Learning and Data Mining (DTU, Spring 2024), completed in two assignments by Group 510. The project explores the UCI Wine dataset — a chemical analysis of wines from three cultivars grown in the same region of Italy — through exploratory data analysis, dimensionality reduction, regression, and classification.

## Authors
- Simon Fogh Kristiansen
- Christian Schultz-Nielsen
- Benjamin Chambaudet

## Repository Contents
- Assignment_1_Group510.pdf — Data description, feature summary statistics, exploratory visualization, and PCA analysis
- Assignment2_report_group510.pdf — Regression, classification, model comparison, and statistical evaluation
- Script/...

## Dataset
13 physicochemical attributes (alcohol content, malic acid, ash, alkalinity of ash, magnesium, phenols, flavanoids, color intensity, hue, proline, etc.) measured across 178 wine samples from 3 cultivars. The dataset has no missing or corrupted values.

## Assignment 1 — Exploration & PCA
- **Dataset description**: attribute types (mostly continuous/ratio, with the cultivar label as a discrete/nominal target), summary statistics (mean, std, median, range) for each attribute
- **Visualization**: box plots (raw and standardized) to check for outliers, histograms to assess normality, a full 14×14 scatterplot matrix, and a correlation heatmap
- **Key findings**:
    - Flavanoids and total phenols are the most correlated attributes (r ≈ 0.86)
    - Ash and OD280/OD315 are essentially uncorrelated (r ≈ 0.00)
    - Proline dominates the raw-scale box plots, motivating standardization before further analysis
- **PCA**: standardized data via SVD; 8 principal components are needed to explain 90% of the variance; component loadings and 2D projections (PC1–PC4) show reasonable separation between the three cultivars

## Assignment 2 — Regression & Classification

### Regression
- **Target**: predicting wine color intensity from the other 12 attributes
- **Method**: regularized linear regression with λ tuned over a logarithmic grid (1 to 100,000); optimal λ ≈ 10
- **Model comparison**: two-level cross-validation (K1 = 10 outer, K2 = 5 inner folds) comparing regularized linear regression, an ANN (1–5 hidden units), and a baseline (mean predictor)
- **Statistical evaluation**: paired comparisons (setup I) show both linear regression and the ANN significantly outperform the baseline (p < 0.001), but the difference between linear regression and the ANN is not statistically significant (p ≈ 0.54)

### Classification
- **Target**: predicting the cultivar (3-class) from all attributes
- **Models compared**: multinomial (logistic) regression, ANN, and a majority-class baseline, again using two-level cross-validation
- **Statistical evaluation**: correlated t-test (setup II) shows both multinomial regression and the ANN significantly outperform the baseline; the difference between the ANN and multinomial regression is not statistically significant
  
### Discussion
- Both tasks confirm the dataset is well-suited to classification into the three cultivars, and only moderately complex, so the ANN doesn't show a clear edge over simpler regularized/linear models
- Suggested extensions: fine-tuning the ANN architecture, ensembling (bagging/boosting/stacking), and more training data (only 178 samples in total)

## Prior Use of the Dataset

The dataset has previously been used to benchmark classifiers such as LDA, Regularized Discriminant Analysis (RDA), and 1-NN, and to illustrate discriminant-analysis appreciation functions (see references in the reports).

## References

Full citations are listed at the end of each report, including the original UCI repository entry and prior classification studies on this dataset.
