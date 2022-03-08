import os
import pickle
import pandas as pd
import numpy  as np
from flask import Flask, request, Response
from healthinsurance.HealthInsurance import HealthInsurance


print('Loading predictor model...')
path  = '/home/leandro/repos/PA_04_Leandro/health_insurance_app/models/'
model = pickle.load(open(path + 'model_linear_regression.pkl','rb'))
print('Loaded!')


print("\n========= INITIALIZING SERVER ==========\n\n")

#initialize api
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def healthinsurance_predict():
    test_json = request.get_json()
    
    if test_json:
        if isinstance(test_json,dict): #unique example
            test_raw = pd.DataFrame(test_json,index=[0])
        else: #multiple examples
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
        
        #instantiate HealthInsurance class
        pipeline = HealthInsurance()

        #data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        print(df1)
        
        #feature Engeneering
        df2 = pipeline.feature_engeneering(df1)
        print(df2)
        
        #data preparation
        df3 = pipeline.data_preparation(df2)
        print(df3)
        
        #prediction
        df_response = pipeline.get_prediction(model,test_raw ,df3)
        
        return df_response
    else:
        return Response('{}',status=200, mimetype='application/json')
    

if __name__ =='__main__':
	port = os.environ.get('PORT',5000)
	app.run('0.0.0.0',port=port)
	#


