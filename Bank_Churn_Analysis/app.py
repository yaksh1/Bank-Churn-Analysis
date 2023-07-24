import pandas as pd
import streamlit as st
import pickle
import lightgbm as lgb


df_1=pd.read_csv("Churn-Records.csv")

# load the saved model
filename = 'lgb_model_smote.sav'
model = pickle.load(open(filename,'rb'))


# creating a function to predict
def churn_prediction(data):
  new_df = pd.DataFrame(data, columns = ['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember',
                                         'EstimatedSalary','Satisfaction Score','Card Type','Point Earned'])
  

  df_2 = pd.concat([df_1, new_df], ignore_index = True) 

  # create bins of Estimated Salary
  labels = ['0-50000','51000-100000','100001-150000','150001-200000']
  df_2['EstimatedSalary_group'] = pd.cut(df_2.EstimatedSalary, bins=[0,50000,100000,150000,200000], labels=labels)
  
  # Create bins of credit score
  labels = ['350-450','451-550','551-650','651-850']
  df_2['CreditScore_group'] = pd.cut(df_2.CreditScore, bins=[349,450,550,650,851], labels=labels)
  
  # Group the Age in bins
  labels = ['18-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100']
  df_2['Age_group'] = pd.cut(df_2.Age, bins=[17,30,40,50,60,70,80,90,100], labels=labels)
    
  #drop column Age
  df_2.drop(columns= ['Age'], axis=1, inplace=True)
  df_2.head()   
  
  
  
  
  new_df__dummies = pd.get_dummies(df_2[['CreditScore','Geography','Gender','Balance','NumOfProducts',
                                         'EstimatedSalary','Satisfaction Score','Card Type','Point Earned','EstimatedSalary_group','CreditScore_group','Age_group']])

  new_df__dummies['Tenure'] = df_2['Tenure']
  new_df__dummies['IsActiveMember'] = df_2['IsActiveMember']
  new_df__dummies['HasCrCard'] = df_2['HasCrCard']
  new_df__dummies.Tenure = pd.to_numeric(new_df__dummies.Tenure, errors='coerce',downcast='integer')
  new_df__dummies.IsActiveMember = pd.to_numeric(new_df__dummies.IsActiveMember, errors='coerce',downcast='integer')
  new_df__dummies.HasCrCard = pd.to_numeric(new_df__dummies.HasCrCard, errors='coerce',downcast='integer')
  
  # return st.dataframe(new_df__dummies.Tenure)
  
  single = model.predict(new_df__dummies.tail(1))
  probablity = model.predict_proba(new_df__dummies.tail(1))[:,1]
  
  if single==1:
      o1 = "This customer is likely to be churned!!"
      o2 = "Confidence: {}".format(probablity*100)
      return o1 + " " + o2 
  else:
      o1 = "This customer is likely to continue!!"
      o2 = "Confidence: {}".format(probablity*100)
      return o1 + " " + o2 
      
def main():
  
  # title
  st.title('Customer Churn Prediction')

  # getting input data from the user
  CreditScore = st.number_input('CreditScore')
  Geography = st.selectbox('Geography',('France','Germany','Spain'))
  Gender = st.selectbox('Gender',('Female','Male'))
  Age = st.number_input('Age')
  Tenure = st.selectbox('Tenure',('0','1','2','3','4','5','6','7','8','9','10'))
  Balance = st.number_input('Balance')
  NumOfProducts = st.number_input('NumOfProducts')
  HasCrCard = st.selectbox('HasCrCard',('1','0'))
  IsActiveMember = st.selectbox('IsActiveMember',('1','0'))
  EstimatedSalary = st.number_input('EstimatedSalary')
  SatisfactionScore = st.selectbox('Satisfaction Score',(1,2,3,4,5))
  CardType = st.selectbox('Card Type',('DIAMOND','SILVER','GOLD','PLATINUM'))
  PointEarned = st.number_input('Point Earned')
  

  
  # code for prediction
  prediction = ''
  
  # creating a button for prediction
  if st.button("Predict:"):
    prediction = churn_prediction([[CreditScore,Geography,Gender,Age, Tenure, Balance, NumOfProducts,
          HasCrCard, IsActiveMember, EstimatedSalary, SatisfactionScore,CardType,
          PointEarned]])
    
  st.success(prediction)
  

if __name__ == '__main__':
  main()