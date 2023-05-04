import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


class LifeExpectancy:
    def __init__(self):
        st.set_page_config(
            page_title="Ex-stream-ly Cool App",
            page_icon="🧊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        try:
            self.df = pd.read_csv('~/Downloads/Life-Expectancy-Data-Updated.csv')
        except FileNotFoundError as e:
            st.error(f'File Not Found: {e}')
        else:
            self.main()

    def main(self):
        st.header('Predict Life Expectancy')

        lrm = self.train_model()

        self.predict_life_expectancy(lrm)

        st.header('Exploratory Data Analysis')

        # Displaying buttons for user interaction
        st.write('\n')
        data_analysis_button, year_life_expectancy_button, gdp_life_button, distribution_life_button, \
        correlation_button = st.columns(5)

        if data_analysis_button.button('Data Analysis'):
            self.exploratory_data_analysis()

        if year_life_expectancy_button.button('Year-Life Expectancy'):
            self.year_life_expectancy()

        if gdp_life_button.button('GDP-Life Expectancy'):
            self.gdp_life_expectancy()

        if distribution_life_button.button('Distribution of Life Expectancy'):
            self.distribution_life_expectancy()

        if correlation_button.button('Correlation Variables'):
            self.correlation_variables()

    def exploratory_data_analysis(self):
        st.subheader('First 5 Rows')
        st.write(self.df.head())

        st.subheader('Last 5 Rows')
        st.write(self.df.tail())

        st.subheader('Shape')
        rows, cols = self.df.shape
        st.write('Rows: ', rows)
        st.write('Columns: ', cols)

        st.subheader('Column Names')
        st.write(self.df.columns)

        st.subheader('Data Types')
        st.write(self.df.dtypes)

        st.subheader('Summary Statistics')
        st.write(self.df.describe())

        st.subheader('Unique Values')
        st.write(self.df.nunique())

        st.subheader('Missing Values')
        st.write(self.df.isna().sum())

    def year_life_expectancy(self):
        fig, axis = plt.subplots()
        sns.lineplot(data=self.df, x='Year', y='Life_expectancy', hue='Region', ax=axis)
        sns.set(style='darkgrid', palette='Pastel1')
        st.pyplot(fig, use_container_width=False)

    def gdp_life_expectancy(self):
        fig, axis = plt.subplots()
        sns.scatterplot(data=self.df, x='GDP_per_capita', y='Life_expectancy', hue='Region', ax=axis)
        sns.set(style='darkgrid')
        st.pyplot(fig, use_container_width=False)

    def distribution_life_expectancy(self):
        fig, axis = plt.subplots()
        sns.histplot(data=self.df, x='Life_expectancy', kde=True)
        sns.set(style='darkgrid', palette='Pastel1')
        st.pyplot(fig, use_container_width=False)

    def correlation_variables(self):
        fig, axis = plt.subplots()
        sns.heatmap(self.df.corr(), annot=False, cmap='coolwarm')
        sns.set(style='darkgrid', palette='Pastel1')
        st.pyplot(fig, use_container_width=False)

    def train_model(self):
        # Splitting data into features and target
        X = self.df[['Infant_deaths', 'Adult_mortality', 'BMI', 'Population_mln', 'GDP_per_capita', 'Schooling',
                     'Economy_status_Developed']]
        y = self.df['Life_expectancy']

        # Splitting data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

        # Fitting the model
        lrm = LinearRegression()
        lrm.fit(X_train, y_train)

        # Making predictions on test set
        predictions = lrm.predict(X_test)

        # Evaluating model performance
        score = metrics.explained_variance_score(y_test, predictions)

        # Displaying predictions and model performance
        st.write(f'Model Performance: {score * 100: .2f}%')

        return lrm

    def collect_user_inputs(self):
        input_columns = ['Infant_deaths', 'Adult_mortality', 'BMI', 'Population_mln', 'GDP_per_capita', 'Schooling',
                         'Economy_status_Developed']
        user_input = []
        for column in input_columns:
            col = column.replace('_', ' ')
            try:
                user_input.append(st.text_input(f'Enter {col}'))
            except ValueError:
                st.error(f'Invalid input for {col}. Please enter a number.')
                return None

        # Creating input dataframe from user inputs
        input_data = dict(zip(input_columns, user_input))
        input_df = pd.DataFrame([input_data])

        # Convert all columns to numeric data type
        input_df = input_df.apply(pd.to_numeric, errors='coerce')

        return input_df

    def predict_life_expectancy(self, lrm):
        # Collecting user inputs
        input_df = self.collect_user_inputs()

        if input_df is None:
            # Error handling for invalid inputs
            st.error('Invalid input. Please enter valid numerical values.')

        else:
            prediction_button = st.button('Predict')

            if prediction_button:
                try:
                    # Making prediction on user input
                    prediction = lrm.predict(input_df)

                    # Displaying prediction on user input
                    st.write(f'Prediction on User Input: **{prediction[0]:.1f}**')

                except Exception as e:
                    # Error handling for model prediction
                    st.error('Prediction failed. Please check your inputs and try again.')


if __name__ == '__main__':
    le = LifeExpectancy()
