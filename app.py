
import pandas as pd
import numpy as np
import streamlit as st
import pickle 
import sklearn 


model = pickle.load(open('GBCModel.dat', 'rb'))
Xdata = pd.read_pickle("FinalX.pkl")

def main():
	st.title('Afib or Normal ECG Classifier')
	st.subheader('Perform these steps before uploading the test/csv file -')
	if uploaded_file ==None:
		return st.error(X)
	uploaded_file = st.file_uploader('Upload the text/csv file.')
	 
	df1= pd.read_csv(uploaded_file, header =None).T

	
	new_df =  pd.concat([Xdata,df1],axis=0, ignore_index=True)
	new_df1 = new_df.fillna(new_df.mean())

	X_set = new_df1[176:]

	if  st.button("Click Here to predict"):
		result = model.predict(X_set)
		result_list = (["ECG graph corresponds to Normal" if x == [0] else "ECG graph corresponds to Afib" for x in result] )
		test = pd.DataFrame(result_list , columns= ['ECG Result'])
		prob = np.round((model.predict_proba(X_set)*100),3)
		predictdf = pd.DataFrame(prob, columns= ['Probability of ECG being Normal % ','Probability of ECG graph being Afib % '] )
		finaldf = pd.concat([test,predictdf],axis=1)
		st.write(finaldf)

if __name__ ==  '__main__':
    main() 
