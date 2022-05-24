from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open('model_pickle','rb'))

wine = pd.read_csv("static/wine_dataset_sheet.csv")
print(wine)

@app.route('/wine-test', methods=["POST", "GET"])
def wine_test():
    wine_type = (wine['type'].unique())
    fixed_acidity = sorted(wine['fixed acidity'].unique())
    volatile_acidity = sorted(wine['volatile acidity'].unique())
    citric_acid = sorted(wine['citric acid'].unique())
    residual_sugar = sorted(wine['residual sugar'].unique())
    chlorides = sorted(wine['chlorides'].unique())
    free_sulfur_dioxide = sorted(wine['free sulfur dioxide'].unique())
    total_sulfur_dioxide = sorted(wine['total sulfur dioxide'].unique())
    density = sorted(wine['density'].unique())
    pH = sorted(wine['pH'].unique())
    sulphates = sorted(wine['sulphates'].unique())
    alcohol = sorted(wine['alcohol'].unique())

    return render_template('wine-test.html', wine_type=wine_type, fixed_acidity=fixed_acidity,
                    volatile_acidity=volatile_acidity, citric_acid=citric_acid, residual_sugar=residual_sugar, 
                    chlorides=chlorides, free_sulfur_dioxide=free_sulfur_dioxide,
                    total_sulfur_dioxide=total_sulfur_dioxide, 
                    density=density, pH=pH, sulphates=sulphates, alcohol=alcohol)

@app.route('/predict', methods=['POST'])
def predict():
    wine= str(request.form.get('wine'))
    fixed_acidity = float(request.form.get('fixed_acidity'))
    volatile_acidity = float(request.form.get('volatile_acidity'))
    citric_acid = float(request.form.get('citric_acid'))
    residual_sugar = float(request.form.get('residual_sugar'))
    chlorides = float(request.form.get('chlorides'))
    free_sulfur_dioxide = float(request.form.get('free_sulfur_dioxide'))
    total_sulfur_dioxide = float(request.form.get('total_sulfur_dioxide'))
    density = float(request.form.get('density'))
    pH = float(request.form.get('pH'))
    sulphates = float(request.form.get('sulphates'))
    alcohol = float(request.form.get('alcohol'))
    print ("Wine=",wine,"FXD=",fixed_acidity)

    if wine == "white":
        wine=1
    else:
        wine=0

    input_data=(wine,fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol)
    print ("INPUT_DATA = ",input_data)
    input_data_as_numpy_array=np.asarray(input_data)
    print ("input_data_as_numpy_array=", input_data_as_numpy_array)
    #reshape the data as we are predicting the label for once one instance 
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    print ("input_data_reshaped=", input_data_reshaped)
    
    
    prediction = model.predict(input_data_reshaped)
    
    print ("prediction=",prediction)

    if prediction[0]==1:
        print ('Good Quality Wine')
        prediction="Good Quality Wine"
    else:
        print ('Bad Quality Wine')
        prediction="Bad Quality Wine"
    
    return str(prediction)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/wine-knowledge')
def wineknowledge():
    return render_template('wine-knowledge.html')


@app.route('/blog')
def blog():
    return render_template('blog.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/team')
def team():
    return render_template('team.html')

if __name__ == "__main__":
    app.run(debug=True)