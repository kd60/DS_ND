# import libraries 
import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine

def load_data(titanic_filepath):
    """
    Loads  datasets
    
    Parameters:
    titanic_filepath: titanic csv file
    
    Returns:
    raw_data: dataframe containing titanic_filepath 

    """
    # load datasets
    raw_data = pd.read_csv(titanic_filepath)

    return raw_data

def clean_data(raw_data):
    """
    Cleans the dataframe.
    
    Parameters:
    df: DataFrame
    
    Returns:
    df: Cleaned DataFrame
    
    """
    raw_data.isna().sum()
    print("The Cabin column is missing", sum(raw_data['Cabin'].isna()), "values out of",len(raw_data['Cabin']))
    clean_data = raw_data.drop('Cabin', axis=1)
    median_age = raw_data["Age"].median()
    clean_data["Age"] = clean_data["Age"].fillna(median_age)
    clean_data["Embarked"] = clean_data["Embarked"].fillna('U')
    gender_columns = pd.get_dummies(clean_data['Sex'], prefix='Sex')
    embarked_columns = pd.get_dummies(clean_data["Embarked"], prefix="Pclass")
    preprocessed_data = pd.concat([clean_data, gender_columns], axis=1)
    preprocessed_data = pd.concat([preprocessed_data, embarked_columns], axis=1)
    preprocessed_data = preprocessed_data.drop(['Sex', 'Embarked'], axis=1)
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    categorized_age = pd.cut(preprocessed_data['Age'], bins)
    preprocessed_data['Categorized_age'] = categorized_age
    preprocessed_data = preprocessed_data.drop(["Age"], axis=1)
    cagegorized_age_columns = pd.get_dummies(preprocessed_data['Categorized_age'], prefix='Categorized_age')
    preprocessed_data = pd.concat([preprocessed_data, cagegorized_age_columns], axis=1)
    preprocessed_data = preprocessed_data.drop(['Categorized_age'], axis=1)

    return clean_data

    # ///////////////////////////


def preprocessed_data(clean_data):
        """
        processing the dataframe.
        
        Parameters:
        df: DataFrame
        
        Returns:
        df: processed DataFrame
        
        """
        gender_columns = pd.get_dummies(clean_data['Sex'], prefix='Sex')
        embarked_columns = pd.get_dummies(clean_data["Embarked"], prefix="Pclass")
        preprocessed_data = pd.concat([clean_data, gender_columns], axis=1)
        preprocessed_data = pd.concat([preprocessed_data, embarked_columns], axis=1)
        preprocessed_data = preprocessed_data.drop(['Sex', 'Embarked'], axis=1)
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
        categorized_age = pd.cut(preprocessed_data['Age'], bins)
        preprocessed_data['Categorized_age'] = categorized_age
        preprocessed_data = preprocessed_data.drop(["Age"], axis=1)
        cagegorized_age_columns = pd.get_dummies(preprocessed_data['Categorized_age'], prefix='Categorized_age')
        preprocessed_data = pd.concat([preprocessed_data, cagegorized_age_columns], axis=1)
        preprocessed_data = preprocessed_data.drop(['Categorized_age'], axis=1)

        class_survived = preprocessed_data[['Pclass', 'Survived']]

        first_class = class_survived[class_survived['Pclass'] == 1]
        second_class = class_survived[class_survived['Pclass'] == 2]
        third_class = class_survived[class_survived['Pclass'] == 3]

        print("In first class", sum(first_class['Survived'])/len(first_class)*100, "% of passengers survived")
        print("In second class", sum(second_class['Survived'])/len(first_class)*100, "% of passengers survived")
        print("In third class", sum(third_class['Survived'])/len(first_class)*100, "% of passengers survived")
        categorized_pclass_columns = pd.get_dummies(preprocessed_data['Pclass'], prefix='Pclass')
        preprocessed_data = pd.concat([preprocessed_data, categorized_pclass_columns], axis=1)
        preprocessed_data = preprocessed_data.drop(['Pclass'], axis=1)
        preprocessed_data = preprocessed_data.drop(['Name', 'Ticket', 'PassengerId'], axis=1)
                
        return preprocessed_data
    # ///////////////////////////////
    

   
    
def save_data(df, database_filepath):
    """Stores df in a SQLite database."""
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('titanic', engine, index=False, if_exists='replace')  



def main():
    """Loads data, cleans data, saves data to database"""
    if len(sys.argv) == 3:

        titanic_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    Titanic: {}\n    '
              .format(titanic_filepath))
        df = load_data(titanic_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print('processing data...')
        df=preprocessed_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
        

    
    else:
        print('Please provide the filepaths of the dataset ')


if __name__ == '__main__':
    main()
