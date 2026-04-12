# Face Detection & Recognition System

A MATLAB-based face detection and recognition system comparing multiple machine learning approaches across different feature extraction methods. Group project for CSC3067 Vision and Machine Learning at Queen's University Belfast.

## Overview

The system classifies face vs non-face images using four ML algorithms, each tested with multiple feature representations. A sliding window detector is also implemented for localising faces within full images.

## Algorithms Implemented

| Algorithm | Feature Types Tested |
|---|---|
| KNN | Full image pixels, HOG, Gabor filters, LBP |
| SVM | Full image pixels, HOG, Gabor filters, LBP |
| Deep Neural Network | Full image pixels, HOG |
| Random Forest | HOG, Gabor filters, LBP |

## Feature Extraction

- **Full Image** — raw pixel values as feature vector
- **HOG (Histogram of Oriented Gradients)** — edge/gradient based descriptor
- **Gabor Filters** — texture-based frequency/orientation features
- **LBP (Local Binary Patterns)** — local texture descriptor

## Evaluation Methodology

Each classifier was evaluated using:
- Standard 50/50 train-test split
- 60/40 train-test split
- K-fold cross-validation

Results logged in `*_Log.txt` files alongside each script.

## Project Structure

```
├── KNN/          # K-Nearest Neighbours classifiers
├── SVM/          # Support Vector Machine classifiers
├── Deep Neural Network/   # DNN with sliding window detector
├── RF/           # Random Forest classifiers
```

## Requirements

- MATLAB (tested with R2024b)
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox (for DNN)

## Notes

This was a group project. The codebase covers systematic comparison of ML approaches for face detection, with each algorithm explored across multiple feature engineering strategies.
