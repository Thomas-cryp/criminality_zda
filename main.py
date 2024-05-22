import os
import pandas as pd
import geopandas as gpd
from shapely.ops import nearest_points
from shapely.geometry import Point
from correlation import DistrictCorrelation


def read_lookup_table(file_path, column_name):
    df = pd.read_csv(file_path)
    df.rename(columns={'id': column_name, 'label': f'{column_name}_label'}, inplace=True)
    return df


def get_district_name(x, y, districts_gdf, district_label_dict):
    point = Point(x, y)

    containing_district = districts_gdf[districts_gdf.geometry.contains(point)]

    if not containing_district.empty:
        district_id = containing_district.iloc[0]['id']
        district_name = district_label_dict.get(district_id, "District not found")

        return district_name
    else:
        districts_gdf = districts_gdf.to_crs("EPSG:5514")

        districts_gdf['centroid'] = districts_gdf.geometry.centroid

        multipoint = districts_gdf['centroid'].unary_union

        nearest_geometry = nearest_points(point, multipoint)

        nearest_district = districts_gdf.loc[districts_gdf.centroid == nearest_geometry[1]]

        district_id = nearest_district.iloc[0]['id']
        district_name = district_label_dict.get(district_id, "District not found")

        return district_name


class DataParser:
    def __init__(self):
        self.main_file_dir = 'data/22_to_23'
        self.relevance_file_path = 'data/relevance.csv'
        self.state_file_path = 'data/states.csv'
        self.types_file_path = 'data/types.csv'
        self.districts_shapefile_path = 'data/shp/cz.shp'
        self.district_label_df = pd.read_csv('data/districts.csv')
        self.types_edit_df = pd.read_csv('data/types_edit.csv')
        self.types_set = set(self.types_edit_df['name'])
        self.result_name = 'data/parsed_data.csv'
        self.all_dfs = []

    def select_type(self, types):
        matching_types = [t for t in types if t in self.types_set]
        if len(matching_types) == len(types):
            return 'Delete'
        else:
            return types

    def run(self):
        all_dfs = []

        print('Parsing data...')
        for file_name in os.listdir(self.main_file_dir):
            if file_name.endswith('.csv'):
                main_file_path = os.path.join(self.main_file_dir, file_name)
                main_df = pd.read_csv(main_file_path, delimiter=',')

                relevance_df = read_lookup_table(self.relevance_file_path, 'relevance')
                state_df = read_lookup_table(self.state_file_path, 'state')

                types_df = pd.read_csv(self.types_file_path)
                types_df.rename(columns={'id': 'types', 'name': 'types_name'}, inplace=True)

                types_df = types_df[['types', 'types_name']]

                main_df = main_df.merge(relevance_df, how='left', left_on='relevance', right_on='relevance')
                main_df = main_df.merge(state_df, how='left', left_on='state', right_on='state')
                main_df = main_df.merge(types_df, how='left', left_on='types', right_on='types')

                main_df.drop(columns=['relevance', 'state', 'types'], inplace=True)
                main_df.rename(columns={
                    'relevance_label': 'relevance',
                    'state_label': 'state',
                    'types_name': 'types'
                }, inplace=True)

                districts_gdf = gpd.read_file(self.districts_shapefile_path)

                main_df['district'] = main_df.apply(lambda row: get_district_name(row['x'], row['y'], districts_gdf, dict(zip(self.district_label_df.id, self.district_label_df.label))), axis=1)

                main_df['date'] = pd.to_datetime(main_df['date'], utc=True, errors='coerce')
                main_df['month'] = main_df['date'].dt.month
                main_df['year'] = main_df['date'].dt.year
                main_df['time'] = main_df['date'].dt.round('h').dt.strftime('%H:00')

                if 'mp' in main_df.columns:
                    main_df.drop(columns=['mp'], inplace=True)

                if 'x' in main_df.columns:
                    main_df.drop(columns=['x'], inplace=True)

                if 'y' in main_df.columns:
                    main_df.drop(columns=['y'], inplace=True)

                if 'date' in main_df.columns:
                    main_df.drop(columns=['date'], inplace=True)

                all_dfs.append(main_df)

        final_df = pd.concat(all_dfs, ignore_index=True)

        final_df = final_df.groupby('id').agg({
            'year': 'first',
            'month': 'first',
            'time': 'first',
            'district': 'first',
            'relevance': 'first',
            'state': 'first',
            'types': lambda x: list(set(x))
        }).reset_index()

        final_df['time'] = pd.to_datetime(final_df['time'], format='%H:%M')
        final_df['time'] = final_df['time'].dt.round('h')
        final_df['time'] = final_df['time'].dt.strftime('%H:00')

        final_df['types'] = final_df['types'].apply(self.select_type)
        final_df = final_df[final_df['types'] != 'Delete']

        district_label = pd.read_csv('data/districts.csv')

        district_label = district_label.melt(id_vars=['id', 'label'], var_name='year', value_name='popularity')
        district_label['year'] = district_label['year'].astype(int)

        final_df = final_df.merge(district_label, how='left', left_on=['district', 'year'], right_on=['label', 'year'])
        final_df['popularity'] = final_df['popularity'].astype('category')

        final_df.rename(columns={'id_x': 'id'}, inplace=True)
        final_df.to_csv(self.result_name, index=False, columns=['id', 'year', 'month', 'time', 'district', 'relevance', 'state', 'types', 'popularity'])
        print('Data parsed and saved to parsed_data.csv')


def main():
    data_parser = DataParser()
    data_parser.run()

    print('Processing data...')
    data_file_path = 'data/parsed_data.csv'
    popularity_file_path = 'data/districts.csv'
    district_correlation = DistrictCorrelation(data_file_path, popularity_file_path)
    for year in [2023, 2022]:
        district_correlation.calculate_correlation_and_regression(year)
        district_correlation.display_results(year)
        district_correlation.plot_regression(year)
        district_correlation.create_yearly_csvs(year)
    district_correlation.create_summary_csv()
    print('Data processed')


if __name__ == "__main__":
    main()







