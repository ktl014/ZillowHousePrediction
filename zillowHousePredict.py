'''
zillowHousePredict

Created on Jan 17 2018 21:27 
#@author: Kevin Le 
'''
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

def main():
    # EDA
    # - find number of column datatypes
    # Clean up data --> get rid of missing values, find number of missing values
    # Split data into train and test for before after Oct 15 2016
    # Regression --> random forest classification???
    # Feature importance
    print('\nProcessing data for csv files ...')
    filename = ['/Users/ktl014/PycharmProjects/PersonalProjects/ZillowHousePrediction/properties_2017.csv',
                '/Users/ktl014/PycharmProjects/PersonalProjects/ZillowHousePrediction/properties_2016.csv',
                '/Users/ktl014/PycharmProjects/PersonalProjects/ZillowHousePrediction/train_2016_v2.csv',
                '/Users/ktl014/PycharmProjects/PersonalProjects/ZillowHousePrediction/train_2017.csv']
    train16_df = pd.read_csv(filename[2], parse_dates=['transactiondate'])
    exploreData(train16_df)

def exploreData(train16_df):
    #LogError EDA
    plt.close('all')
    fig = plt.figure(figsize=(12,8))
    fig.add_subplot(211)
    plt.scatter(range(train16_df.shape[0]), np.sort(train16_df['logerror'].values))
    plt.xlabel('index', fontsize=12)
    plt.ylabel('logerror', fontsize=12)
    plt.title('Logerror Scatterplot')

    fig.add_subplot(212)
    plt.xlim(-1.0,1.0)
    # plt.hist(np.sort(train16_df['logerror'].values), bins=50)
    sns.distplot(train16_df.logerror.values, bins=50, kde=False)
    plt.xlabel('logerror', fontsize=12)
    plt.title('Logerror Histogram')
    plt.tight_layout()
    plt.show()
    plt.savefig('results/LogErrorEDA.png')



if __name__ == '__main__':
    main()
