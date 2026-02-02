# Logistic Regression from Scratch using Gradient Descent

This project implements **Logistic Regression** without using any machine learning libraries such as scikit-learn.  
The goal is to understand how **binary classification models learn** using **Gradient Descent** and the **Sigmoid function**.

---

## Problem Statement

Given student scores:

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

Where:

0 → Fail  
1 → Pass  

The model must learn a **decision boundary** that separates Pass and Fail cases  
by minimizing classification error — **without hard-coding any rules**.

---

## Learning Algorithm

The model uses:

- Sigmoid function for probability estimation  
- Binary classification with thresholding  
- Batch Gradient Descent for optimization  

Update rules:

w = w - lr * dw  
b = b - lr * db  

Where gradients are computed manually.

---

## What this project demonstrates

- How sigmoid converts linear scores into probabilities  
- Why logistic loss is suitable for classification  
- How gradient descent updates decision boundaries  
- How thresholding turns probabilities into class labels  

---

## How to Run

```bash
python logistic_regression.py
```

Author: Ishaan Sharma
