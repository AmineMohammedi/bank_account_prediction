import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('Financial_inclusion_dataset.csv')

lb = LabelEncoder()

df_clean = df.copy()

df_clean['country'] = lb.fit_transform(df['country'])
df_clean['bank_account'] = lb.fit_transform(df['bank_account'])
df_clean['location_type'] = lb.fit_transform(df['location_type'])
df_clean['cellphone_access'] = lb.fit_transform(df['cellphone_access'])
df_clean['gender_of_respondent'] = lb.fit_transform(df['gender_of_respondent'])
df_clean['relationship_with_head'] = lb.fit_transform(df['relationship_with_head'])
df_clean['marital_status'] = lb.fit_transform(df['marital_status'])
df_clean['education_level'] = lb.fit_transform(df['education_level'])
df_clean['job_type'] = lb.fit_transform(df['job_type'])

df_clean = df_clean.drop('uniqueid', axis=1)

def iqr_fence(x):
    Q1 = x.quantile(0.25)
    Q3 = x.quantile(0.75)
    IQR = Q3 - Q1
    Lower_Fence = Q1 - (1.5 * IQR)
    Upper_Fence = Q3 + (1.5 * IQR)
    u = max(x[x < Upper_Fence])
    l = min(x[x > Lower_Fence])
    return [u, l]

iqr_fence(df_clean['age_of_respondent'])
iqr_fence(df_clean['household_size'])

df_clean = df_clean.drop(df_clean[df_clean['age_of_respondent'] > 83].index)
df_clean = df_clean.drop(df_clean[df_clean['age_of_respondent'] < 16].index)
df_clean = df_clean.drop(df_clean[df_clean['household_size'] > 9].index)
df_clean = df_clean.drop(df_clean[df_clean['household_size'] < 1].index)

x = df_clean[['age_of_respondent', 'household_size', 'country', 'location_type', 'cellphone_access', 'gender_of_respondent', 'relationship_with_head', 'marital_status', 'education_level', 'job_type']]
y = df_clean['bank_account']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42, criterion='gini', max_depth=6, min_samples_split=5, min_samples_leaf=4, splitter='best')

clf.fit(x_train, y_train)


def load_data():
    df = pd.read_csv('Financial_inclusion_dataset.csv')
    for col in df.columns:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])
    return df

def train_model(df):
    X = df.drop(columns=['bank_account', 'year', 'uniqueid'])
    y = df['bank_account']  
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(df['bank_account'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier() 
    model.fit(X_train, y_train)
    return model

def main():
    st.title('Streamlit Check Point 2')
    
    # Load data
    data_load_state = st.text('Loading data...')
    df = load_data()
    data_load_state.text('Data loaded successfully!')
    

    # Train model
    model_load_state = st.text('Training model...')
    model = train_model(df)
    model_load_state.text('Model trained successfully!')

    # Input features
    countries = ['Kenya', 'Rwanda', 'Tanzania', 'Uganda']
    selected_country = st.selectbox("Countries: ", countries)
    location_type = st.radio("Select Your Location Type: ", ['Rural','Urban'])
    cellphone_access = st.checkbox("Cellphone Access")
    cellphone_access = 1 if cellphone_access else 0
    household_size = st.slider("Select the Number of people living in one house", 1, 21)
    age_of_respondent = st.slider("Select Your age", 16, 83)
    gender_of_respondent = st.radio("Select Your Gender: ", ['Male','Female'])
    gender_of_respondent = 1 if gender_of_respondent == 'Female' else 0
    relationship_with_head = st.selectbox("Relationship with the head of the house: ",['Spouse', 'Head of Household', 'Other relative', 'Child', 'Parent','Other non-relatives'])
    marital_status = st.selectbox("The martial status: ",['Married/Living together', 'Widowed', 'Single/Never Married','Divorced/Seperated', 'Dont know'])
    education_level = st.selectbox("The Highest level of education: ",['Secondary education', 'No formal education',
                                                          'Vocational/Specialised training', 'Primary education','Tertiary education', 'Other/Dont know/RTA'])
    job_type = st.selectbox("The Type of your job has: ",['Self employed', 'Government Dependent',
       'Formally employed Private', 'Informally employed',
       'Formally employed Government', 'Farming and Fishing',
       'Remittance Dependent', 'Other Income',
       'Dont Know/Refuse to answer', 'No Income'])
    
    # Encode categorical features
    label_encoder = LabelEncoder()
    selected_country_encoded = label_encoder.fit_transform([selected_country])[0]
    relationship_with_head_encoded = label_encoder.fit_transform([relationship_with_head])[0]
    marital_status_encoded = label_encoder.fit_transform([marital_status])[0]
    education_level_encoded = label_encoder.fit_transform([education_level])[0]
    job_type_encoded = label_encoder.fit_transform([job_type])[0]
    location_type_encoded = label_encoder.fit_transform([location_type])[0]

    # Prediction
    if st.button('Predict'):
        X_user = [[age_of_respondent, household_size, selected_country_encoded, location_type_encoded, cellphone_access, gender_of_respondent, relationship_with_head_encoded, marital_status_encoded, education_level_encoded, job_type_encoded]]
        prediction = model.predict(X_user)
        if prediction[0] == 1:
            st.success('You have a bank account')
        else:
            st.error('You dont have a bank account')

if __name__ == '__main__':
    main()
