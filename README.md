
# Breast Cancer Prediction using Ensemble Techniques

- Breast cancer represents one of the diseases that make a high number of deaths every year. It is the most common type of all cancers and the main cause of women's deaths worldwide.


- Breast cancer is a significant health concern, and early detection plays a crucial role in improving patient outcomes.

- This project addresses the challenge of accurately predicting breast cancer using machine learning techniques. 








## Aim objective


- This project aims to predict breast cancer in patients by utilizing ensemble techniques. The dataset contains information about cell nuclei taken from breast masses, and based on the provided features, the model predicts whether a patient has benign or malignant breast cancer.


 - The project involves exploratory data analysis, model building using ensemble techniques, and evaluating the model's performance based on various metrics.
## Dataset Information

-  The dataset comprises several predictor variables and one target variable, Diagnosis. The target variable has two classes: 'Benign,' indicating that the cells are not harmful or cancerous, and 'Malignant,' indicating that the patient has cancer and the cells have a harmful effect. 

-  The predictor variables include features such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. 

-  Additionally, the dataset includes mean, standard error, and "worst" values for these features, resulting in a total of 30 features.
## Variable Description


The columns in the dataset are described as follows:

- Radius: Mean of distances from center to points on the perimeter.
- Texture: Standard deviation of gray-scale values.
- Perimeter: Observed perimeter of the lump
- Area: Observed area of the lump
- Smoothness: Local variation in radius lengths
- Compactness: Perimeter^2 / area - 1.0
- Concavity: Severity of concave portions of the contour
- Concave points: Number of concave portions of the contour
- Symmetry: Lump symmetry
- Fractal dimension: "Coastline approximation" - 1
- Diagnosis: Whether the patient has cancer or not ('Malignant' or 'Benign')
## Installation

To run the project, you will need the following:

1. Google Colab.
2. Required Python packages (numpy, pandas,scikit-learn, matplotlib, etc.)

```bash
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix


```
    
## Project Flow

#### The project comprises the following tasks:


1. Exploratory Data Analysis (EDA):
 - Perform EDA process to gain insights into the dataset. Analyze the distributions, correlations, and summary statistics of the features. Using Seaborn or Matplotlib visualize the data as plots and graphs  to understand the relationships between variables.

2. Data Pre-processing: 
- Handle missing values, if any replace with mean value, and performed  data pre-processing steps such as feature scaling . Prepare the dataset for model training by encoding categorical variables,
#### Benign ----------> no cancer as (0)
#### Malignant -------> cancer as (1) 
and splitting into training and testing sets.

3. Ensemble Model Development: 
- Build an ensemble model using techniques such as Random Forest, Gradient Boosting, or AdaBoost.  Combine multiple models in ensemble learning and enhance predictive accuracy.

4. Model Evaluation:
- Evaluate the model's performance using performance metrics other than accuracy. Interpret precision, recall, F1-score  and confusion matrix values to measure the model's effectiveness in correctly classifying benign and malignant cases.

5. Hyperparameter Tuning:
-  Optimize the hyperparameters of the ensemble model to improve its performance. Hyperparameters techniques such as Gridsearchcv coupled with cross-validation to identify the best hyperparameter values.

6. Interpretation and Conclusion:
-  Interpret the results obtained from the ensemble model. Analyze the importance of different features in predicting breast cancer . Deploy the model in Streamlit and user can enter the corresponding feature values and get the optimize results.




## Learning Outcome

#### By working on this project, I have learned the following :

1. Understanding the Dataset:
- By working on this project, i have gained experience in analyzing and exploring the provided dataset and  how to interpret and understand the variables, their relationships, and the significance of the features in predicting breast cancer.

2. Data Pre-processing: 
- Learned,various data pre-processing techniques required to prepare the dataset for model training. This includes handling missing values, scaling features, and other necessary data transformations such as encoding data.

3. Ensemble Techniques:
-  The project focuses on using ensemble techniques such as Random Forest, Gradient Boosting, or AdaBoost to build a predictive model.
- Got a clear understanding of ensemble learning and how it can improve prediction accuracy compared to individual models.

4. Model Evaluation Metrics: 
- In addition to accuracy, learned how  to evaluate the model's performance using various metrics such as precision, sensitivity, specificity, and AUC-ROC.

5. Hyperparameter Tuning:
-  The project includes hyperparameter tuning using Gridsearch with cross-validation techniques. Understand a better way of knowing how to optimize the model's hyperparameters to achieve better performance and avoid overfitting.

6. Interpreting Results:
-  By analyzing the results of the model,  gain insights into the predictive power of the ensemble techniques used.
-  Deployment has been done by using Streamlit app so that , user can feed an input to the model and get prediction from the model. 

Overall, this project will enhance in  understanding of breast cancer prediction using ensemble techniques, feature importance, model evaluation, and hyperparameter tuning. It will also help to improve the skills in data analysis, data pre-processing, and building robust machine learning models.


## Results and Performance Evaluation

- The project aims to achieve accurate breast cancer predictions using ensemble techniques. 
- The results will be evaluated based on metrics such as accuracy, precision, recall, F1 score and confusion matrix. 
- By analyzing the performance metrics, we can determine the model's effectiveness in identifying benign and malignant cases.






## Acknowledgements
 - The dataset used in this project was sourced from guvi  and is used for educational purposes only.  We    acknowledge and thank the contributors of the dataset for making it publicly available.


- For more details and a better understanding of the project, please refer to the Google Colab file (BreastCancerPrediction.ipynb) and (breastcancer.py) provided in this repository.

Thanks!

