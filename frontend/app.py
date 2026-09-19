import streamlit as st, pandas as pd, numpy as np, requests, os
st.set_page_config(page_title='SuperKart Sales Forecast',layout='wide')
st.title('SuperKart Sales Revenue Forecasting')
URL=os.getenv('BACKEND_URL','http://127.0.0.1:7860')
def call(rows):
 r=requests.post(URL+'/predict',json=rows,timeout=120); r.raise_for_status(); return r.json()['predictions']
def prep(d):
 d=d.copy()
 if 'Product_Id' in d and 'Product_Id_char' not in d: d['Product_Id_char']=d['Product_Id'].astype(str).str[:2]
 if 'Store_Establishment_Year' in d and 'Store_Age_Years' not in d: d['Store_Age_Years']=pd.Timestamp.today().year-d['Store_Establishment_Year']
 if 'Product_Type' in d and 'Product_Type_Category' not in d:
  p={'Dairy','Meat','Fruits and Vegetables','Frozen Foods','Bread','Seafood'}; d['Product_Type_Category']=np.where(d['Product_Type'].isin(p),'Perishables','Non Perishables')
 return d
t1,t2=st.tabs(['Single Prediction','Batch Prediction'])
with t1:
 with st.form('f'):
  w=st.number_input('Product Weight',value=10.0); s=st.selectbox('Product Sugar Content',['Low Sugar', 'No Sugar', 'Regular', 'reg']); a=st.number_input('Product Allocated Area',value=.05); m=st.number_input('Product MRP',value=100.0); z=st.selectbox('Store Size',['High', 'Medium', 'Small']); city=st.selectbox('City Type',['Tier 1', 'Tier 2', 'Tier 3']); typ=st.selectbox('Store Type',['Departmental Store', 'Food Mart', 'Supermarket Type1', 'Supermarket Type2']); fam=st.selectbox('Product ID Family',['DR', 'FD', 'NC']); age=st.number_input('Store Age',value=10); cat=st.selectbox('Product Category',['Perishables','Non Perishables']); ok=st.form_submit_button('Predict Sales')
 if ok:
  row={'Product_Weight':w,'Product_Sugar_Content':s,'Product_Allocated_Area':a,'Product_MRP':m,'Store_Size':z,'Store_Location_City_Type':city,'Store_Type':typ,'Product_Id_char':fam,'Store_Age_Years':age,'Product_Type_Category':cat}
  try: st.success(f'Predicted Sales Revenue: {call([row])[0]:,.2f}')
  except Exception as e: st.error(str(e))
with t2:
 f=st.file_uploader('Upload Batch_Data_SuperKart.csv',type=['csv'])
 if f:
  d=pd.read_csv(f); st.dataframe(d.head())
  if st.button('Run Batch Prediction'):
   try:
    pred=call(prep(d).to_dict(orient='records')); out=d.copy(); out['Predicted_Product_Store_Sales_Total']=pred; st.dataframe(out); st.download_button('Download Predictions',out.to_csv(index=False).encode(),'SuperKart_Batch_Predictions.csv','text/csv')
   except Exception as e: st.error(str(e))
