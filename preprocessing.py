import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

def categorizeAge(age):
    if age < 13:
        return "Child"
    if age < 24:
        return "Youth"
    if age < 64:
        return "Adult"
    return "Senior"


def createNewFeatures(titanic_df):
    titanic_df["Family size"] = titanic_df["Number of parents/children"] + titanic_df["Number of siblings/spouses"] + 1 # To count the passenger
    titanic_df["Discrete age"] = pd.cut(titanic_df["Age"],bins=range(0,85,5))
    titanic_df["Categorized age"] = titanic_df["Age"].apply(categorizeAge)
    titanic_df["Categorized age"] = titanic_df["Categorized age"].astype("category")

def expandFare(titanic_df):
    ticket_values = titanic_df["Ticket"].value_counts()
    
    def ticketCount(ticket):
        return ticket_values[ticket]
    
    titanic_df["Ticket size"] = titanic_df["Ticket"].apply(ticketCount)
    titanic_df["Individual fare"] = titanic_df["Fare"] / titanic_df["Ticket size"]
    
def renameDataCategories(titanic_df):
    new_economic_status_names = {1:"Upper",2:"Middle",3:"Lower"}
    titanic_df["Economic status"].cat.rename_categories(new_economic_status_names,inplace=True)
    new_port_names = {"C":"Cherbourg","Q":"Queenstown","S":"Southhampton"}
    titanic_df["Embarked"].cat.rename_categories(new_port_names,inplace = True)
    new_sex = {"male":"Male","female":"Female"}
    titanic_df["Sex"].cat.rename_categories(new_sex,inplace = True)
    
    
def fillMissingValues(titanic_df):
    titanic_df["Cabin"].fillna("-",inplace=True)
    
    titanic_df['Surname'] = titanic_df['Name'].str.split(', ', expand=True)[0]
    titanic_df['Title'] =  titanic_df['Name'].str.split(', ', expand=True)[1].str.split('. ', expand=True)[0]
    
    null_age = titanic_df["Age"].isnull()
    
    titanic_df["Completed age"] = null_age
    
    age_by_title = titanic_df.groupby(by="Title")["Age"].agg("mean")
    
    age_for_nan = age_by_title[titanic_df.loc[null_age,"Title"]]
    
    age_for_nan.index = titanic_df[null_age].index
    
    titanic_df.loc[null_age,"Age"] = age_for_nan
    
    titanic_df.loc[uncommon_titles,"Title"] = "Other"
    titanic_df['Title'] = titanic_df['Title'].astype("category")
    

    
def prepareTestDataset(test_df):
    expandFare(test_df)
    fillMissingValues(test_df)
    createNewFeatures(test_df)
    renameDataCategories(test_df)
    
    
####

def normalize(numericalData):
    return (numericalData - numericalData.mean()) / numericalData.std()

def ordinalEncoder(datos_a_codificar):
    encoder = OrdinalEncoder()
    return encoder.fit_transform(datos_a_codificar)

def oneHotEncoder(datos_a_codificar):
    encoder = OneHotEncoder(drop='first', sparse=False)
    return encoder.fit_transform(datos_a_codificar)

def encodeDataset(titanic_df,categorical_columns,numerical_columns):
    categoricalDataEncoded = oneHotEncoder(titanic_df[categorical_columns])
    numericalData = titanic_df[numerical_columns]
    data = np.hstack((np.array(numericalData), categoricalDataEncoded))
    return data

def encodeAndNormalizeData(titanic_df,categorical_columns,numerical_columns):
    categoricalDataEncoded = oneHotEncoder(titanic_df[categorical_columns])
    numericalData = titanic_df[numerical_columns]
    numericalData = normalize(numericalData)
    data = np.hstack((np.array(numericalData), categoricalDataEncoded))
    return data

    
