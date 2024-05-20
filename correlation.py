import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


class DistrictCorrelation:
    def __init__(self, data_file_path, popularity_file_path):
        self.data = pd.read_csv(data_file_path)
        self.popularity_data = pd.read_csv(popularity_file_path)
        self.popularity_data = self.popularity_data.melt(id_vars=['id', 'label'], var_name='year', value_name='popularity')
        self.popularity_data['year'] = pd.to_numeric(self.popularity_data['year'])
        self.correlation = {}
        self.regression_result = {}
        self.merged_data = {}
        self.yearly_data = {}

    def calculate_correlation_and_regression(self, year):
        data_year = self.data[self.data['year'] == year]

        grouped = data_year.groupby('district').agg(
            num_records=('id', 'count')
        ).reset_index()

        popularity_year = self.popularity_data[self.popularity_data['year'] == year][['label', 'popularity']].rename(columns={'label': 'district'})

        self.merged_data[year] = pd.merge(grouped, popularity_year, on='district')
        self.correlation[year] = self.merged_data[year]['num_records'].corr(self.merged_data[year]['popularity'])
        self.regression_result[year] = stats.linregress(self.merged_data[year]['num_records'], self.merged_data[year]['popularity'])

    def plot_regression(self, year):
        if year in self.regression_result and year in self.merged_data:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.merged_data[year]['num_records'], self.merged_data[year]['popularity'], label='Data Points')
            plt.plot(
                self.merged_data[year]['num_records'],
                self.regression_result[year].slope * self.merged_data[year]['num_records'] + self.regression_result[year].intercept,
                color='red',
                label='Fitted Line'
            )
            plt.xlabel('Number of Records')
            plt.ylabel('Popularity')
            plt.title(f'Linear Regression: Number of Records vs. Popularity ({year})')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'lin_reg_{year}.png')
        else:
            print(f'Regression results or merged data for {year} not available. Please run the calculation first.')

    def create_yearly_csvs(self, year):
        grouped = self.data[self.data['year'] == year].groupby('district').size().reset_index(name='num_records')
        popularity_year = self.popularity_data[self.popularity_data['year'] == year]
        merged = pd.merge(grouped, popularity_year, how='left', left_on='district', right_on='label')
        merged[['district', 'num_records', 'popularity']].to_csv(f'district_data_{year}.csv', index=False)
        self.yearly_data[year] = merged[['district', 'num_records', 'popularity']]

    def create_summary_csv(self):
        summary_data = pd.DataFrame()
        for year, data in self.yearly_data.items():
            data = data.rename(columns={'num_records': f'num_records_{year}'})
            if summary_data.empty:
                summary_data = data
            else:
                summary_data = pd.merge(summary_data, data, on='district', how='outer')

        summary_data.to_csv('summary_data.csv', index=False)

    def display_results(self, year):
        if year in self.correlation:
            print(f'Correlation for {year}: {self.correlation[year]}')
        if year in self.regression_result:
            print(f'Linear Regression Results for {year}:')
            print(f'  Slope: {self.regression_result[year].slope}')
            print(f'  Intercept: {self.regression_result[year].intercept}')
        else:
            print(f'Results for {year} have not been calculated yet.')

