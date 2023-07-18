#flask, scikit-learn, pandas, pickle-mixin
import pandas as pd
import pickle
from flask import Flask, render_template, request
import os

app = Flask(__name__)

csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cleaned_data.csv')
data = pd.read_csv(csv_file_path)
model_file_path = os.path.join(os.path.dirname(__file__), 'RidgeModel.pkl')
pipe = pickle.load(open(model_file_path, 'rb'))


@app.route('/')
def index():
    
    mainroads = sorted(data['mainroad'].unique())
    return render_template('/index.html', mainroads=mainroads)

@app.route('/predict' , methods=['POST'])
def predict():
    print("i'm in predict")
    if request.method == 'POST':
        stry = request.form.get('stories')
        ar = request.form.get('area')
        bds = request.form.get('bedroom')
        bth = request.form.get('bathroom')
        mainrd = request.form.get('mainroad')
        furnishingstat = request.form.get('furnishingstatus')
        
        print(stry,ar,bds,bth,mainrd,furnishingstat)
        input = pd.DataFrame([[stry, ar,bds,bth,mainrd,furnishingstat]],columns=['stories','area','bedrooms','bathrooms','mainroad','furnishingstatus'])
        prediction = pipe.predict(input)[0]
        
        
        return str(prediction)
    

if __name__ == "__main__":
    app.run(debug=True, port=5001)
