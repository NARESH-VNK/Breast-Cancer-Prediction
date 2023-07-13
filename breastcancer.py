import pandas as pd
import pip
import numpy as np
import streamlit as st
import sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix




df = pd.read_csv("cancer.csv")

# drop unnecessary columns
df.drop(['Unnamed: 32','id'],axis =1,inplace= True)


#Encoing the categorical data
encoder = OrdinalEncoder()
df['diagnosis'] = encoder.fit_transform(df[['diagnosis']])


# Split the dependent(X) and independent variable(y)
X = df.drop(['diagnosis'],axis=1)
y = df['diagnosis']

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


options = st.sidebar.selectbox('MENU',['ABOUT','APPROACH','PREDICTION','CONCLUSION'])

if options == "ABOUT":
    st.title("INTODUCTION")
    st.write("- ##### Breast cancer represents one of the diseases that make a high number of deaths every year. It is the most common type of all cancers and the main cause of women's deaths worldwide.")
    st.write("- ##### Breast cancer is a significant health concern, and early detection plays a crucial role in improving patient outcomes.")
    st.write("- ##### This project addresses the challenge of accurately predicting breast cancer using machine learning techniques.")
    st.title("AIM")
    st.write("- ##### This project aims to predict breast cancer in patients by utilizing ensemble techniques. The dataset contains information about cell nuclei taken from breast masses, and based on the provided features, the model predicts whether a patient has benign or malignant breast cancer.")
    st.write("- ##### The project involves exploratory data analysis, model building using ensemble techniques, and evaluating the model's performance based on various metrics.")

if options =="APPROACH":
    st.subheader("Exploratory Data Analysis (EDA)")
    st.write("- Perform EDA process to gain insights into the dataset. Analyze the distributions, correlations, and summary statistics of the features. Using Seaborn or Matplotlib visualize the data as plots and graphs to understand the relationships between variables.")
    st.subheader("Data Pre-Processing")
    st.write("- Handle missing values,if any replace with mean value, and performed data pre-processing steps such as feature scaling.Prepare the dataset for model training by encoding categorical variables\
    Benign --------> no cancer as (0)\
    Malignant ------> cancer as (1)\
    and splitting into training and testing sets.")

    st.subheader("Ensemble Model Development")
    st.write("- Build an ensemble model using techniques such as Random Forest, Gradient Boosting, or AdaBoost. Combine multiple models in ensemble learning and enhance predictive accuracy.")
    st.subheader(" Model Evaluation")
    st.write("- Evaluate the model's performance using performance metrics other than accuracy. Interpret precision, recall, F1-score and confusion matrix values to measure the model's effectiveness in correctly classifying benign and malignant cases.")
    st.subheader('Hyperparameter Tuning')
    st.write('- Optimize the hyperparameters of the ensemble model to improve its performance. Hyperparameters techniques such as Gridsearchcv coupled with cross-validation to identify the best hyperparameter values.')
    st.subheader("Interpretation and Conclusion")
    st.write("- Interpret the results obtained from the ensemble model. Analyze the importance of different features in predicting breast cancer . Deploy the model in Streamlit and user can enter the corresponding feature values and get the optimize results.")


if options == "PREDICTION":

    st.title("Breast Cancer Prediction")
    st.markdown("### Enter all the corresponding values")

    col1, col2, col3,col4,col5,col6 = st.columns(6)
    with col1:
        radius_mean  = st.number_input('Enter radius_mean value')

    with col2:
        texture_mean = st.number_input('Enter texture_mean value')
    with col3:
        perimeter_mean = st.number_input('Enter perimeter_mean value')
    with col4:
        area_mean = st.number_input('Enter area_mean value')
    with col5:
        smoothness_mean = st.number_input('Enter smoothness_mean value ')
    with col6:
        compactness_mean = st.number_input('Enter compactness_mean value')





    col7, col8, col9,col10,col11,col12 = st.columns(6)
    with col7:
        concavity_mean = st.number_input('Enter concavity_mean value ')
    with col8:
        concave_points_mean= st.number_input('Enter concave points_mean value')
    with col9:
        symmetry_mean = st.number_input('Enter symmetry_mean value ')
    with col10:
        fractal_dimension_mean = st.number_input('Enter fractal_dimension_mean value')
    with col11:
        radius_se = st.number_input('Enter radius_se value')
    with col12:
        texture_se = st.number_input('Enter texture_se value')




    col13, col14, col15,col16,col17,col18 = st.columns(6)
    with col13:
        perimeter_se = st.number_input("Enter perimeter_se value")
    with col14:
        area_se = st.number_input("Enter area_se value")
    with col15:
        smoothness_se = st.number_input("Enter smoothness_se value")
    with col16:
        compactness_se = st.number_input("Enter compactness_se value")
    with col17:
        concavity_se = st.number_input("Enter concavity_se value")
    with col18:
        concave_points_se = st.number_input("Enter concave_points_se value")


    col19,col20,col21,col22,col23,col24 = st.columns(6)


    with col19:
        symmetry_se = st.number_input("Enter symmetry_se value")
    with col20:
        fractal_dimension_se = st.number_input("Enter fractal_dimension_se value")
    with col21:
        radius_worst = st.number_input("Enter radius_worst value")
    with col22:
        texture_worst = st.number_input("Enter texture_worst value")
    with col23:
        perimeter_worst = st.number_input("Enter perimeter_worst value")
    with col24:
        area_worst = st.number_input("Enter area_worst value")





    col25,col26,col27,col28,col29,col30 = st.columns(6)

    with col25:
        smoothness_worst = st.number_input("Enter smoothness_worst value")
    with col26:
        compactness_worst = st.number_input("Enter compactness_worst value")
    with col27:
        concavity_worst = st.number_input("Enter concavity_worst value")
    with col28:
        concave_points_worst = st.number_input("Enter concave_points_worst value")
    with col29:
        symmetry_worst = st.number_input("Enter symmetry_worst value")
    with col30:
        fractal_dimension_worst = st.number_input("Enter fractal_dimension_worst value")


    #scaling to  standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)



    #  Random Forest classifier
    model = RandomForestClassifier(max_depth=None,min_samples_split=2, n_estimators=50,random_state=42)
    result_rf = model.fit(X_train, y_train)

    # Make predictions on the new test data
    newtestdata = [radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]
    scaler = StandardScaler()
    new_test_data = np.array([newtestdata])
    X_train = scaler.fit_transform(X_train)
    new_test_data_scale = scaler.transform(new_test_data)

    new_scaled_prediction = result_rf.predict(new_test_data_scale)


    #
    # st.markdown([12.45, 15.7, 82.57, 477.1, 0.1278, 0.17, 0.1578, 0.08089, 0.2087, 0.07613, 0.3345, 0.8902, 2.217, 27.19, 0.00751, 0.03345, 0.03672, 0.01137,\
    #  0.02165, 0.005082, 15.47, 23.75, 103.4, 741.6, 0.1791, 0.5249, 0.5355, 0.1741, 0.3985, 0.1244])

    if st.button("Predict"):

        if new_scaled_prediction == 0:
            st.markdown("### The Predicted Outcome is 0 ")
            st.write("### The Cells are Benign, They are not Harmful or There is no Breast Cancer")
        else:
            st.markdown("### The Predicted Outcome is 1")
            st.write("### The Cells are Malignant, They have a  Harmful effect or the Patient has a Breast Cancer!")

if options =="CONCLUSION":
    st.subheader("Results and Performance Evaluation")
    st.write("- The project aims to achieve accurate breast cancer predictions using ensemble techniques.The results will be evaluated based on metrics such as accuracy, precision, recall, F1 score and confusion matrix.")
    st.write("- By analyzing the performance metrics, we can determine the model's effectiveness in identifying benign and malignant cases.")
    st.subheader("Conclusion")
    st.write("- This project contributes to the field of breast cancer prediction by utilizing ensemble techniques and exploring various performance evaluation metrics. By accurately predicting breast cancer cases, medical professionals can make informed decisions about diagnosis and treatment, potentially improving patient outcomes.")
    st.write("- The knowledge gained from this project enhances our understanding of ensemble techniques and their applicability in medical data analysis.")