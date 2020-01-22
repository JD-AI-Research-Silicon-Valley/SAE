import json
from time import time
import pandas as pd
from pandas import DataFrame
from fastparquet import write as parquet_writer
from fastparquet import ParquetFile as parquet_loader
from fastparquet.api import ParquetFile

def save_dataframe2json(dataframe, filename, orient='records'):
    with open(filename, "w") as output_file:
        output_file.write(dataframe.to_json(orient=orient))

def save_dataframe2json2(dataframe, filename, orient='records'):
    dataframe.to_json(filename, orient=orient)

def save_dataframe2pickle(dataframe, filename):
    dataframe.to_pickle(filename)

def loadPickleAsDataFrame(pickle_file_name):
    dataframe = pd.read_pickle(pickle_file_name)
    return dataframe

def loadJSonAsDataFrame(jsonFileName):
    """
    :param jsonFileName:
    :return:
    """
    start = time()
    with open(jsonFileName, 'r') as f:
        json_data = json.load(f)
        dataFrame = pd.DataFrame(json_data)
    print("Json data {} loading takes {} seconds".format(dataFrame.shape[0], time() - start))
    return dataFrame

def save_dataframe2Parquet(dataframe, filename):
    parquet_writer(filename, dataframe)


def loadParquetAsDataFrame(fileName) -> DataFrame:
    start = time()
    pf = parquet_loader(fileName)
    print("Parquet data loading takes {} seconds".format(time() - start))
    dataframe = pf.to_pandas()
    print("Parquet to pandas takes {} seconds".format(time() - start))
    return dataframe

def loadParquet(fileName) -> ParquetFile:
    start = time()
    pf = parquet_loader(fileName)
    print("Parquet data loading takes {} seconds".format(time() - start))
    return pf

if __name__ == '__main__':

    import sys
    print(sys.path)
