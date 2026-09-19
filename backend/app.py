from flask import Flask,request,jsonify
import pandas as pd, numpy as np, joblib
app=Flask(__name__)
model=joblib.load('superkart_best_model.joblib')
FEATURE_COLUMNS=['Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area', 'Product_Type', 'Product_MRP', 'Store_Id', 'Store_Size', 'Store_Location_City_Type', 'Store_Type', 'Product_Id_char', 'Store_Age_Years', 'Product_Type_Category']
P={'Dairy','Meat','Fruits and Vegetables','Frozen Foods','Bread','Seafood'}
def prep(df):
 df=df.copy()
 if 'Product_Id' in df and 'Product_Id_char' not in df: df['Product_Id_char']=df['Product_Id'].astype(str).str[:2]
 if 'Store_Establishment_Year' in df and 'Store_Age_Years' not in df: df['Store_Age_Years']=pd.Timestamp.today().year-df['Store_Establishment_Year']
 if 'Product_Type' in df and 'Product_Type_Category' not in df: df['Product_Type_Category']=np.where(df['Product_Type'].isin(P),'Perishables','Non Perishables')
 for c in ['Product_Id','Store_Establishment_Year','Product_Store_Sales_Total']:
  if c in df: df=df.drop(columns=[c])
 for c in FEATURE_COLUMNS:
  if c not in df: df[c]=np.nan
 return df[FEATURE_COLUMNS]
@app.get('/')
def home(): return jsonify({'status':'running','service':'SuperKart Sales Forecast Backend'})
@app.post('/predict')
def predict():
 try:
  x=request.get_json(force=True); x=[x] if isinstance(x,dict) else x
  return jsonify({'predictions':model.predict(prep(pd.DataFrame(x))).tolist()})
 except Exception as e: return jsonify({'error':str(e)}),400
if __name__=='__main__': app.run(host='0.0.0.0',port=7860)
