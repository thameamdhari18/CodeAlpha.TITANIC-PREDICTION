# CodeAlpha.TITANIC-PREDICTION

README: Titanic Survival Prediction
Author: Thameam Dhari
Batch: April
Domain: Data Science
Aim :
The aim of this project is to build a model that predicts whether a passenger on the Titanic survived or not based on given features.

Dataset :
The dataset for this project is imported from a CSV file, "archive.zip". The dataset contains information about passengers on the Titanic, including their survival status, class (Pclass), sex (Gender), and age (Age).

Libraries Used : 
The following important libraries were used for this project:

*numpy
*pandas
*matplotlib.pyplot
*seaborn
*sklearn.preprocessing.LabelEncoder
*sklearn.model_selection.train_test_split
*sklearn.linear_model.LogisticRegression
Data Exploration and Preprocessing :
The dataset was loaded using pandas as a DataFrame, and its shape and a glimpse of the first 10 rows were displayed using df.shape and df.head(10).
Descriptive statistics for the numerical columns were displayed using df.describe() to get an overview of the data, including missing values.
The count of passengers who survived and those who did not was visualized using sns.countplot(x=df['Survived']).
The count of survivals was visualized with respect to the Pclass using sns.countplot(x=df['Survived'], hue=df['Pclass']).
The count of survivals was visualized with respect to the gender using sns.countplot(x=df['Sex'], hue=df['Survived']).
The survival rate by gender was calculated and displayed using df.groupby('Sex')[['Survived']].mean().
The 'Sex' column was converted from categorical to numerical values using LabelEncoder from sklearn.preprocessing.
After encoding the 'Sex' column, non-required columns like 'Age' were dropped from the DataFrame.
Model Training
The feature matrix X and target vector Y were created using relevant columns from the DataFrame.
The dataset was split into training and testing sets using train_test_split from sklearn.model_selection.
A logistic regression model was initialized and trained on the training data using LogisticRegression from sklearn.linear_model.
Model Prediction
The model was used to predict the survival status of passengers in the test set.
The predicted results were printed using log.predict(X_test).
The actual target values in the test set were printed using Y_test.
A sample prediction was made using log.predict([[2, 1]]) with Pclass=2 and Sex=Male(1)
