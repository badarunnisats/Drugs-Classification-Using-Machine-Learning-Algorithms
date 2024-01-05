# Drugs Classification Using Machine Learning Algorithms

![Screenshot 2024-01-05 165234](https://github.com/badarunnisats/Drugs-Classification-Using-Machine-Learning-Algorithms/assets/109198401/c87acdf1-a480-444d-a274-03846e53ca6a)

This Github aims to predict insurance charges based on various factors using machine learning models.Build different classification models from historical data of patients and their responses to different medications. Then you use the trained algorithms to predict the class of an unknown patient or to find a proper drug for a new patient.

# Overview

This project aims to predict insurance charges based on various factors using machine learning models.Build different classification models from historical data of patients and their responses to different medications. Then you use the trained algorithms to predict the class of an unknown patient or to find a proper drug for a new patient.

# Project Structure

**1. Data Collection:**

- The dataset is obtained from source link using the wget command.
- The data is read into a Pandas DataFrame for further analysis

```python
#Downloading the data from the provided URL and reading it into a DataFrame
!wget -O drug200.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv
df = pd.read_csv("drug200.csv")
```

**2. Exploratory Data Analysis (EDA):**

- Basic libraries such as NumPy, Pandas, Matplotlib, and Seaborn are imported.
- Descriptive statistics, null checks, and visualizations are performed to understand the data.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

#Descriptive stats
df.describe()

#Null check
df.isnull().value_counts()

#Blood pressure of patients

sns.set_theme(style="darkgrid")
sns.countplot(y="BP", data=df, palette="crest")
plt.ylabel('Blood Pressure')
plt.xlabel('Total')
plt.show()


```

**3. Data Preprocessing:**

- Label encoding is applied to categorical variables (e.g., Drug,Cholestrol,BP,Sex) to convert them into numerical format. from sklearn.preprocessing import LabelEncoder

```python
enc = LabelEncoder()
cdf = df
cdf['Drug'] = enc.fit_transform(df['Drug'])
cdf['Cholesterol'] = enc.fit_transform(df['Cholesterol'])
cdf['BP'] = enc.fit_transform(df['BP'])
cdf['Sex'] = enc.fit_transform(df['Sex'])
```
**4. Feature Selection:**

- Independent (features) and dependent (target) variables are selected.

```python
X = cdf[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = cdf['Drug']
```

**5. Modeling:**

- Various regression models are explored, including Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVR),Naive Bayes,KNN.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)

missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
for col in missing_cols:
    X_test_encoded[col] = 0

rf_classifier = RandomForestClassifier()
svm_classifier = SVC()
nb_classifier = GaussianNB()
knn_classifier = KNeighborsClassifier()
dt_classifier = DecisionTreeClassifier()
lr_classifier = LogisticRegression()

rf_classifier.fit(X_train_encoded, y_train)
svm_classifier.fit(X_train_encoded, y_train)
nb_classifier.fit(X_train_encoded, y_train)
knn_classifier.fit(X_train_encoded, y_train)
dt_classifier.fit(X_train_encoded, y_train)
lr_classifier.fit(X_train_encoded, y_train)

rf_predictions = rf_classifier.predict(X_test_encoded)
svm_predictions = svm_classifier.predict(X_test_encoded)
nb_predictions = nb_classifier.predict(X_test_encoded)
knn_predictions = knn_classifier.predict(X_test_encoded)
dt_predictions = dt_classifier.predict(X_test_encoded)
lr_predictions = lr_classifier.predict(X_test_encoded)

rf_accuracy = accuracy_score(y_test, rf_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)
nb_accuracy = accuracy_score(y_test, nb_predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)
dt_accuracy = accuracy_score(y_test, dt_predictions)
lr_accuracy = accuracy_score(y_test, lr_predictions)

print("Random Forest Accuracy:", rf_accuracy)
print("SVM Accuracy:", svm_accuracy)
print("Naive Bayes Accuracy:", nb_accuracy)
print("KNN Accuracy:", knn_accuracy)
print("Decision Tree Accuracy:", dt_accuracy)
print("Logistic Regression Accuracy:", lr_accuracy)

```
6. Conclusion:

The model with the highest accuracy rate can be considered the best model for classifying the for the future patients having hte same illness in future.
```python
best_model = compare.sort_values(by='Accuracy', ascending=False).iloc[0]
print("Best Model:", best_model['Model'])
print("Accuracy:", best_model['Accuracy'])

```
