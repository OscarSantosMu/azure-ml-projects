# import libraries
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

def main(args):
    # read data
    df = get_data(args.input_data)

    cleaned_data = clean_data(df)

    normalized_data = normalize_data(cleaned_data)

    output_df = normalized_data.to_csv((Path(args.output_data) / "Titanic.csv"), index = False)

# function that reads the data
def get_data(path):
    df = pd.read_csv(path)

    # Count the rows and print the result
    row_count = (len(df))
    print('Preparing {} rows of data'.format(row_count))
    
    return df

# function that removes missing values
def clean_data(df):
    df = df.dropna()
    
    return df

# function that normalizes the data
def normalize_data(df):
    #datos de columna sexo a números
    df['Sex'].replace(['female','male'],[0,1],inplace=True)
    #datos de origen de embarque en números
    df['Embarked'].replace(['Q','S', 'C'],[0,1,2],inplace=True)
    #Reemplazo de datos faltantes en la edad por la media calculada en 30
    promedio = 30
    df['Age'] = df['Age'].replace(np.nan, promedio)
    #Se elimina la columna de "Cabin"
    df.drop(['Cabin'], axis = 1, inplace=True)
    #Elimino las columnas que considero que no son necesarias para el análisis
    df = df.drop(['PassengerId','Name','Ticket'], axis=1)
    #Se elimina las filas con los datos perdidos
    df.dropna(axis=0, how='any', inplace=True)

    return df

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--input_data", dest='input_data',
                        type=str)
    parser.add_argument("--output_data", dest='output_data',
                        type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")