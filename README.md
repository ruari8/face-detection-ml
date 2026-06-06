# Face Detection and Recognition in MATLAB

Coursework group project for CSC3067 Vision and Machine Learning at Queen's University Belfast. The project compares several classical machine learning approaches for classifying face and non-face images, then applies selected models to sliding-window face localisation.

This is a MATLAB coursework codebase. It is useful for showing the modelling experiments and feature engineering, but it is not packaged as a production face recognition application.

## At a glance

| Area | Details |
| --- | --- |
| Project type | Vision and machine learning coursework |
| Language | MATLAB |
| Task | Face vs non-face classification and sliding-window localisation |
| Models | KNN, SVM, Random Forest, shallow neural network |
| Features | Raw pixels, PCA, HOG, Gabor filters, LBP |
| Outputs | Log files, trained `.mat` files, figures, result summaries |

## Methods covered

| Folder | Contents |
| --- | --- |
| `KNN/` | KNN experiments across full-image, PCA, HOG, Gabor, and LBP features |
| `SVM/` | SVM classifiers and sliding-window detector experiments |
| `RF/` | Random Forest models, saved models, detector scripts, figures |
| `Deep Neural Network/` | Shallow neural network experiments and detector scripts |

The experiments compare:

- Standard train/test splits
- 60/40 train/test splits
- K-fold cross-validation
- Multiple feature extraction approaches
- Sliding-window detection with non-maximum suppression

## Repository structure

```text
KNN/
  images/                  Face and non-face image sets
  *.m                      KNN experiment scripts
  *_Log.txt                Previous run logs
  KNN_Results_Summary.csv  Summary of KNN metrics
SVM/
  Sliding Detector/        SVM detector experiments
  archive/                 Older SVM/kernel experiments
RF/
  Sliding Detector/        Random Forest detector experiments
  rf figs/                 Result figures
Deep Neural Network/
  Sliding Detector/        DNN detector experiments
```

## Requirements

- MATLAB, tested during coursework development
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox for the neural network scripts

## Running the experiments

Open the repository in MATLAB and run scripts from the relevant folder so relative dataset paths resolve correctly.

Example KNN run:

```matlab
cd KNN
RunAllScripts
```

Example individual scripts:

```matlab
cd KNN
KNN_HOG

cd ../SVM
SVM_hog

cd ../RF
RF_Hog
```

Sliding-window detector examples are in the `Sliding Detector` subfolders. They use saved model files where present and test images such as `im1.jpg` to `im4.jpg`.

## Results

The repository includes prior run logs and figures. For example, `KNN/KNN_Results_Summary.csv` records KNN accuracy across feature variants, with cross-validation runs generally performing better than the simple train/test splits.

Treat these results as coursework experiment outputs rather than a controlled benchmark. Some folders also contain archived scripts and intermediate files that were kept for traceability.

## Notes

- This was a group project, so file ownership is mixed across the codebase.
- The code is script-oriented MATLAB rather than a package with one unified entry point.
- The included images and generated datasets are intended to reproduce the coursework experiments.
