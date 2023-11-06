from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
def home(request):
    return render(request, 'home.html')
def predict(request):
    return render(request, 'predict.html')
def result(request):
    data = pd.read_csv(r"C:\Users\ayoub\Documents\Python Scripts\MeriSKILL Internship\data\diabetes.csv")
    X = data.drop("Outcome", axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pared = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    result1 = ""
    if pared == [1]:
        result1 = "Positive"
    elif pared == [0]:
        result1 = 'Negative'

    return render(request, "predict.html", {"result2": result1})
