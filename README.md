# CodeAlpha_Disease-Prediction-from-Medical-Data
# Disease Prediction using Machine Learning

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,classification_report



# Load dataset

data = load_breast_cancer()


X=data.data
y=data.target



X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model=RandomForestClassifier(
    n_estimators=100
)


model.fit(
    X_train,
    y_train
)



prediction=model.predict(
    X_test
)



print(
"Accuracy:",
accuracy_score(
    y_test,
    prediction
)
)


print(
classification_report(
    y_test,
    prediction
)
)
