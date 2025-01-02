# Food Image Recognition and Calorie Estimation

This project involves developing a model that accurately recognizes food items from images and estimates their calorie content. The objective is to enable users to track their dietary intake and make informed food choices.

## About the Project

Accurate recognition of food items and estimation of their calorie content from images is a valuable tool for dietary monitoring and health management. This project utilizes machine learning techniques to build a model capable of identifying various food items and estimating their caloric values based on image inputs.

## Dataset

The dataset comprises images of various food items, each labeled with the corresponding food name and calorie content. These images are used to train and evaluate the model.

## Features

- **Data Preprocessing**: Loading images, resizing, normalization, and labeling.
- **Feature Extraction**: Extracting meaningful features from images to serve as input for the model.
- **Model Training**: Training a K-Nearest Neighbors (KNN) classifier on the extracted features.
- **Calorie Estimation**: Estimating the calorie content based on the recognized food item.

## Getting Started

Follow these steps to set up the project locally.

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Jupyter Notebook or JupyterLab
- Necessary Python libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `opencv-python`

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/heemit/PRODIGY_ML_05.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd PRODIGY_ML_05
   ```
  
3. **Install the required packages**:
   ```bash
   pip install numpy pandas matplotlib scikit-learn opencv-python
   ```

## Usage

1. **Prepare the dataset**:   
   - Ensure that the images are organized in a structured format, with each folder representing a food category containing corresponding images.
   
1. **Open the Jupyter Notebook:**   
   ```bash
   jupyter notebook task4.py
   ```

2. **Run the cells sequentially:**
   Execute the cells in sequence to preprocess the data, train the model, and evaluate its performance.

## Model Evaluation

The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. Confusion matrices and classification reports are generated to provide detailed insights into the classifier's effectiveness.
