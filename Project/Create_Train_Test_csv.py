import pandas as pd
from sklearn.model_selection import train_test_split
import click
import torch
from dotenv import find_dotenv, load_dotenv
import logging

@click.command()

@click.option("Trump_path", default = 'Data/raw/', help = 'Path of data')
@click.option("Russian_path", default = 'Data/raw/', help = 'Path of data')
@click.option("data_path", default = 'Data/processed/', help = 'Path of data')

def create_train_test(Trump_path, Russian_path, data_path):
    pd_Russian = pd.read_csv(Trump_path + 'tweets.csv')
    pd_D_T = pd.read_csv(Russian_path + 'realdonaldtrump.csv')
    print(pd_Russian)
    pd_Russian['Label'] = 0
    pd_D_T['Label'] = 1

    pd_Russian.rename(columns = {'text' : 'Tweet'}, inplace=True)
    pd_D_T.rename(columns = {'content' : 'Tweet'}, inplace=True)

    pd_combine = pd.concat([pd_Russian[['Tweet','Label']],pd_D_T[['Tweet','Label']]], ignore_index=True).reset_index(drop=True)

    Train, Test = train_test_split(pd_combine,
                                    random_state=104, 
                                    test_size=0.25, 
                                    shuffle=True,
                                    stratify=pd_combine['Label'])

    Train.to_csv(data_path + 'Train.csv')
    Test.to_csv(data_path + 'Test.csv')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    create_train_test()