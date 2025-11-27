# 🥔 Potato Disease Detection Using Deep Learning

This project implements a deep learning–based system to detect diseases in potato plant leaves using Convolutional Neural Networks (CNN) and transfer learning models like **ResNet50** and **EfficientNetB0**.

The model classifies potato leaf images into multiple disease categories and evaluates performance using **Precision, Recall, and F1 Score**.

---

##  Project Overview

Potato crops are highly vulnerable to leaf diseases, which can significantly reduce yield if not detected early. This project helps farmers and researchers by providing an automated system to classify potato leaf diseases from images.

---

##  Objectives

- Build a deep learning model for potato disease classification.
- Apply transfer learning using ResNet50 and EfficientNetB0.
- Apply image preprocessing and augmentation.
- Evaluate model using accuracy, precision, recall, and F1-score.

---


Each folder contains corresponding potato leaf images.

---

##  Model Architecture

This project includes:
- Custom CNN
- Transfer Learning using:
  - 🧠 ResNet50
  - ⚡ EfficientNetB0

Image preprocessing includes:
- Rescaling
- Random Flip
- Brightness & Contrast Adjustment
- Data Augmentation

---

##  Evaluation Metrics

The model evaluates performance using:

- Accuracy
- Precision
- Recall
- F1 Score

Example metric calculation:

```python
precision = precision_score(val_labels, predicted_classes, average=None)
recall = recall_score(val_labels, predicted_classes, average=None)
f1 = f1_score(val_labels, predicted_classes, average=None)

Requirements

Install the following libraries before running:

pip install tensorflow numpy matplotlib scikit-learn


Dependencies used:

TensorFlow / Keras

NumPy

Matplotlib

Scikit-learn


Results

After training, the model outputs:

Accuracy

Per-class Precision

Per-class Recall

Per-class F1 Score

It also visualizes training and validation accuracy/loss.

🔮 Future Improvements

Add more disease classes.

Increase dataset size.

Deploy as a web or mobile app.

Use attention-based deep learning models.

👨‍💻 Author

Md Ashiqur Rahman
Department of CSE
BSc in Computer Science and Engineering
Specialization: Data Science

