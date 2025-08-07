# Trash Classification with FastAPI and Deep Learning

This project is an end-to-end image classification system designed to identify and categorize trash into different types using a trained deep learning model. The backend is developed using FastAPI, and the frontend is built with HTML, CSS, and JavaScript to provide a responsive and simple user interface.

## Project Overview

The goal of this project is to automate trash classification using a Convolutional Neural Network (CNN) trained on augmented image data. The model predicts the category of trash (e.g., plastic, metal, paper) from an uploaded image. A FastAPI backend serves the model and handles predictions, while the frontend allows users to upload images and view results in real time.

## Features

- Deep learning model trained with data augmentation
- FastAPI backend for serving predictions
- Web interface built with HTML, CSS, and JavaScript
- Real-time prediction and response
- Simple and modular code structure

## Model Details

- Architecture: Convolutional Neural Network (CNN)
- Framework: TensorFlow / Keras
- Augmentation: Rotation, flipping, zoom, etc.
- Dataset: Custom dataset of categorized trash images

Model training notebook: [`Trash_Classification_with_augmentation.ipynb`](./Trash_Classification_with_augmentation.ipynb)

## Tech Stack

| Component    | Technologies                  |
|--------------|-------------------------------|
| Model        | TensorFlow, Keras             |
| Backend      | FastAPI, Uvicorn              |
| Frontend     | HTML, CSS, JavaScript         |
| Image Utils  | Pillow                        |
| Others       | Python, JSON                  |

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trash-classification.git
cd trash-classification
