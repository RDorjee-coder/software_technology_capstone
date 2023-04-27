import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


class LifeExpectancy:
    def __init__(self):
        st.set_page_config(
            page_title="Ex-stream-ly Cool App",
            page_icon="🧊",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://www.extremelycoolapp.com/help',
                'Report a bug': "https://www.extremelycoolapp.com/bug",
                'About': "# This is a header. This is an *extremely* cool app!"
            }
        )

        try:
            self.df = pd.read_csv('~/Downloads/Life-Expectancy-Data-Updated.csv')
        except FileNotFoundError as e:
            st.error(f'File Not Found: {e}')
        else:
            data_analysis_button, year_life_expectancy_button, region_life_button = st.columns(3)

            if data_analysis_button.button('Exploratory Data Analysis'):
                self.exploratory_data_analysis()

            if year_life_expectancy_button.button('Year Life Expectancy'):
                self.year_life_expectancy()

            if region_life_button.button('Region Life Expectancy'):
                self.region_life_expectancy()

    def exploratory_data_analysis(self):
        st.subheader('First 5 Rows')
        st.write(self.df.head())

        st.subheader('Last 5 Rows')
        st.write(self.df.tail())

        st.subheader('Shape')
        st.write(self.df.shape)

        st.subheader('Column Names')
        st.write(self.df.columns)

        st.subheader('Data Types')
        st.write(self.df.dtypes)

        st.subheader('Summary Statistics')
        st.write(self.df.describe())

        st.subheader('Unique Values')
        st.write(self.df.nunique())

    def year_life_expectancy(self):
        fig, axis = plt.subplots()
        sns.barplot(data=self.df, x='Year', y='Life_expectancy', hue='Region', ax=axis)
        sns.set(style='darkgrid', palette='Pastel1')
        st.pyplot(fig, use_container_width=False)

    def region_life_expectancy(self):
        fig, axis = plt.subplots()
        sns.barplot(data=self.df, x='Region', y='Life_expectancy', hue='Region', ax=axis)
        sns.set(style='darkgrid', palette='Pastel1')
        st.pyplot(fig, use_container_width=False)


if __name__ == '__main__':
    le = LifeExpectancy()
