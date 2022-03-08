import pickle
import numpy  as np
import pandas as pd

class HealthInsurance( object ):
    
    def __init__( self ):
        self.home_path = '/home/leandro/repos/PA_04_Leandro/health_insurance_app/features/'
        self.annual_premium_scaler = pickle.load(open (self.home_path + 'annual_premium_scaler.pkl','rb'))
        self.age_scaler = pickle.load(open(self.home_path + 'age_scaler.pkl','rb'))
        self.vintage_scaler = pickle.load(open (self.home_path + 'vintage_scaler.pkl', 'rb'))
        self.target_encode_gender_scaler = pickle.load(open ( self.home_path + 'target_encode_gender_scaler.pkl','rb'))
        self.target_encode_region_code_scaler = pickle.load(open( self.home_path + 'target_encode_region_code_scaler.pkl','rb'))
        self.fe_policy_sales_channel_scaler = pickle.load(open(self.home_path + 'fe_policy_sales_channel_scaler.pkl','rb'))
    
    def data_cleaning(self, df1 ):
        col_names = ['id', 'gender', 'age', 'region_code', 'policy_sales_channel',
                     'previously_insured', 'annual_premium', 'vintage', 'driving_license',
                     'vehicle_damage', 'damage_per_rcode', 'vehicle_age_1-2 Year',
                     'vehicle_age_< 1 Year', 'vehicle_age_> 2 Years', 'response']
        
        df1.columns = col_names
        return df1


        
       
    
    def feature_engeneering(self, df2 ):
        df2['vehicle_damage'] = df2['vehicle_damage'].apply(lambda x: 1 if x=="Yes" else 0)
        #df2['vehicle_age']    = df2['vehicle_age'].apply(lambda x: "over_2years" if x =="> 2 Years" else "between_12years" if x == "1-2 Year" else "below_year")
        #df2['gender']         = df2['gender'].apply(lambda x: x.lower())
        
        return df2
  

    
    def data_preparation( self,df5):
        # Annual Premium
        df5['annual_premium'] = self.annual_premium_scaler.transform( df5[['annual_premium']].values)

        # Age
        df5['age'] = self.age_scaler.transform( df5[['age']].values)
        
        # Vintage
        df5['vintage'] = self.vintage_scaler.transform( df5[['vintage']].values)
        
        ## 5.3 Features Transformation1
    
        df5.loc[ :, 'gender'] = df5['gender'].map( self.target_encode_gender_scaler )
        
        # Region Code- One Hot Encoding / <span class="mark">Target Encoding</span> / Weighted Target Encoding
        #df5.loc[ :, 'region_code'] = df5['region_code'].map ( self.target_encode_region_code_scaler)
       
        # Vehicle Age- One hot encoding / Order encoding / Frequency Encoding 
        df5 = pd.get_dummies( df5, prefix= 'vehicle_age', columns= ['vehicle_age_1-2 Year' ] )
        df5 = pd.get_dummies( df5, prefix= 'vehicle_age', columns= ['vehicle_age_< 1 Year' ] )
        df5 = pd.get_dummies( df5, prefix= 'vehicle_age', columns= ['vehicle_age_> 2 Years' ] )

        # Policy Sales Channel -Target Encoding / Frequency encoding
        #df5.loc[:, 'policy_sales_channel']= df5['policy_sales_channel'].map ( self.fe_policy_sales_channel_scaler )

        df5.fillna(0,inplace = True)

        # Feature Selection
        cols_selected = ['annual_premium', 'vintage', 'age', 'region_code', 'vehicle_damage', 'previously_insured',
                         'policy_sales_channel']
        
        return df5[ cols_selected ]



    def get_prediction( self, model, original_data, test_data ):
        # model prediction
        pred = model.predict_proba( test_data )
        
        # join prediction into original data
        original_data['prediction'] = pred[:,1].tolist()
        
        return original_data.to_json( orient='records', date_format='iso' )
