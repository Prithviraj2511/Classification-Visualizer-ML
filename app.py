from flask import Flask, request, render_template,session
from flask_session import Session
from flask_cors import cross_origin
from base64 import b64encode

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.tree import export_graphviz
from six import StringIO
import pydotplus

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")




@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        pima=session.get("dataframe")
        datasetColumns=session.get("dataframeColumns")
        feature_cols_idx=list(map(int,request.form.get("featuresCols").split(",")))
        output_col_idx=int(request.form.get("outputCol"))
        # print(feature_cols_idx,output_col_idx)
        feature_cols = [datasetColumns[i] for i in feature_cols_idx]
        X = pima[feature_cols]  # Features
        y = pima[datasetColumns[output_col_idx]]  # Target variable

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=1)  # 70% training and 30% test

        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier()
        # Train Decision Tree Classifer
        clf = clf.fit(X_train, y_train)
        # Predict the response for test dataset
        y_pred = clf.predict(X_test)

        accuracy=metrics.accuracy_score(y_test, y_pred)

        dot_data = StringIO()
        export_graphviz(clf, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        # graph.write_png('diabetes.png')
        image=b64encode(graph.create_png()).decode("utf-8")
        return render_template('home.html',accuracy=accuracy,image=image)


    return render_template("home.html",datasetColumns=[])


@app.route("/upload", methods = ["POST"])
@cross_origin()
def upload():
    if request.method == "POST":
        # Create variable for uploaded file
        f = request.files['fileupload']
        # create list of dictionaries keyed by header row
        pima = pd.read_csv(f)
        session["dataframe"]=pima
        datasetColumns = pima.columns
        session["dataframeColumns"]=datasetColumns
        return render_template('home.html',datasetColumns=datasetColumns)


    return render_template("home.html")



if __name__ == "__main__":
    app.run(debug=True)
