import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


class LifeExpectancy:
    def __init__(self):
        try:
            self.df = pd.read_csv('~/Downloads/Life-Expectancy-Data-Updated.csv')
        except FileNotFoundError as e:
            print('File Not Found', e)
        else:
            print(self.df.info())
            print(self.df.describe())
            print(self.df.columns)
            print(self.df.head())

            null_counts = self.df.isnull().sum()
            print('Null values in the dataset: \n', null_counts)

            button_clicked = st.button('year_life_expectancy')

            if button_clicked:
                self.year_life_expectancy()

    def year_life_expectancy(self):
        fig, axis = plt.subplots()
        sns.barplot(data=self.df, x=self.df['Year'], y=self.df['Life_expectancy'], hue='Region', ax=axis)
        print(self.df.groupby(['Region']))
        sns.set(style='darkgrid', palette='Pastel1')
        # plt.show()
        st.pyplot(fig)


if __name__ == '__main__':
    le = LifeExpectancy()
