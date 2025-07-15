### **Osteoporosis Risk Prediction App**  

**Background**  
Osteoporosis is a chronic condition characterized by reduced bone density and increased risk of fractures, particularly in the hip, spine, and wrist. It affects millions of people worldwide and is especially common among older adults and postmenopausal women. Often called a "silent disease," osteoporosis typically goes undiagnosed until a fracture occurs.
Major risk factors include advanced age, female gender, low body mass index (BMI), poor calcium and vitamin D intake, smoking, alcohol use, physical inactivity, chronic diseases, and long-term use of certain medications. Because it often presents no early symptoms, early detection through risk modeling is essential for prevention and management.
This app uses survey data reflecting these risk factors to estimate the probability that an individual has or is at risk for developing osteoporosis. It provides a simple, interactive way for users to upload data, view predictions, and download results.

**How the App Was Built**

1. Data Source and Features
The model was trained on a cleaned dataset (cleaned_survey.csv) containing 54 columns, including demographic, lifestyle, and medical history variables. The target variable is Osteoporosis, indicating whether the respondent had a diagnosis of the condition.

2. Data Preprocessing

Numerical features were imputed using the mean and scaled using StandardScaler.
Categorical features were imputed using the most frequent value and encoded using OneHotEncoder.
All preprocessing steps were wrapped into a ColumnTransformer and integrated into the model pipeline.

3. Model Training
A Random Forest Classifier was used to model the risk of osteoporosis. The model was trained using the preprocessed data and saved using joblib as osteoporosis_model.joblib for efficient reuse during prediction.

4. Streamlit Application
The user interface was built using Streamlit, allowing users to:

View instructions and data requirements in a sidebar
Upload a .csv file containing new survey responses
Preview the uploaded data
Generate and display predictions for osteoporosis risk
Download the results as a CSV file

Streamlit handles the session state and file upload, while the saved model processes input data and returns predicted probabilities for the "Yes" and "No" classes.

**Conclusion**
This app demonstrates how machine learning can be applied to public health and preventive care by enabling fast, data-driven risk assessments for osteoporosis. It provides a valuable tool for early intervention, especially in resource-limited settings or community-based screening programs.
