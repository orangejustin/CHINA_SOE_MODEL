import pandas as pd
from googletrans import Translator
from pandas.api.types import is_numeric_dtype
import numpy as np
import re
import json
from IPython.display import display, Markdown


class soeModel:
    # TODO: parameters name
    # TODO: each table correspond each graph in the paper: needs index
    # TODO: Model table refine
    # TODO: All table updated
    # TODO: Debugs for the error

    # TODO: readable in the notebook.
    # TODO: table, matrixc (varibles) over time (time series table over the years)

    # TODO: Other SOEs.
    # TODO: updated CSVs in folder.

    def __init__(self):
        """
        upload raw data
        """
        # coal types
        self.preprocessed = False
        self.translated = False
        self.correct = False

        power_sale = "raw data/Power Sale.csv"
        power_cost = "raw data/Power Costs.csv"
        elec_cost = "raw data/Electricity Costs.csv"
        # by coal types
        coal_oper = "raw data/Coal Operation Data.csv"
        coal_seg = "raw data/Coal segment operating costs.csv"
        coal_sale = "raw data/Coal sales price.csv"

        self.power_sale = pd.read_csv(power_sale, header=None)
        self.power_cost = pd.read_csv(power_cost, header=None)
        self.elec_cost = pd.read_csv(elec_cost, header=None)
        # new
        self.coal_oper = pd.read_csv(coal_oper, header=None)
        self.coal_seg = pd.read_csv(coal_seg, header=None)
        self.coal_sale = pd.read_csv(coal_sale, header=None)



        # self.df_lst = [self.power_sale, self.power_cost, self.elec_cost]
        self.df_lst = ['power_sale', 'power_cost', 'elec_cost']

    def preprocessing(self):
        """
        Preprocessing of the dataframe
        :param df: pandas DataFrame
        :return: preprocessed dataframe
        """
        for i in self.df_lst:
            df = eval('self.' + i)
            df.loc[0] = df.loc[0].fillna(method='ffill').fillna('')
            df.columns = df.loc[0]
            # power_sale = power_sale.drop(columns=['Remove'])
            df.iloc[1] = df.iloc[1].str.replace('\n', ' ')
            columns = df.iloc[1] + ' ' + df.iloc[0]
            if i == "power_cost" or i == "elec_cost":
                columns = columns.fillna('成本类别')
            df.columns = columns
            df.drop(index=[0, 1], inplace=True)
            df.reset_index(drop=True, inplace=True)
            # replace other symbol of None value to None
            df.replace(['-', '/', '  -   ', ' – '], None, inplace=True)

        return self

    def preprocessing_2(self):
        """
        This method performs the second stage of preprocessing on the DataFrame.

        :return: Returns the instance of the class with the preprocessed DataFrame.
        """
        if self.preprocessed:
            return self
        is_cost = False

        for i in self.df_lst:
            if i == "power_cost" or i == "elec_cost":
                is_cost = True
            df = eval('self.' + i)
            # column names reformatting
            df.loc[0] = df.loc[0].str.extract(r'(\d{4})', expand=False).fillna(method='ffill')
            df = df.fillna('')
            df.iloc[1] = df.iloc[1].str.replace('\n', ' ').str.strip()
            columns = df.iloc[1] + ' ' + df.iloc[0]

            # if the data is a cost-related table
            if is_cost:
                columns = columns.replace(' ', '成本类别')
            df.columns = columns

            # drop the unnecessary rows
            df = df.iloc[2:].reset_index(drop=True)
            df.columns = [col.strip() for col in df.columns]

            # Identify the columns containing a year in the first row
            year_columns = [col for col in df.columns if re.search(r'\d{4}$', col)]
            new_column_names = {col: re.sub(r'\s*\d{4}\s*$', '', col) for col in df.columns}
            # Preprocessing translate to the time-series panel data
            # Purpose is to be splittable afterwards
            for col in year_columns:
                match = re.search(r'\d{4}', col)
                year = match.group(0)
                df[col] = df[col].apply(lambda x: str(x) + '!' + str(year))
                # If Cost Category
                if is_cost:
                    df[col] = df['成本类别'] + '!' + df[col]
                # if power sale
                else:
                    df[col] = df['电厂分类'] + '!' + df['电厂'] + '!' + \
                              df['所在电网'] + '!' + df['地理位置'] + '!' + df[col]

            # Drop and will be adding back afterward
            if is_cost:
                df.drop(['成本类别'], axis=1, inplace=True)
            else:
                df.drop(['电厂分类', '电厂', '所在电网', '地理位置'], axis=1, inplace=True)

            # Rename the cok
            df.rename(columns=new_column_names, inplace=True)
            # Drop the unnecessary columns
            if not is_cost:
                df.drop(['预测毛利润 (百万元)', '计算I38中的加权平均售电电价', '计算Q38中的加权平均售电电价'], axis=1,
                        inplace=True)
            # Get unique column names
            unique_columns = df.columns.unique()

            # Create an empty DataFrame to store the concatenated columns
            concatenated_df = pd.DataFrame()

            # Loop through unique column names and concatenate the columns vertically
            for unique_col in unique_columns:
                # Create an empty list to store values
                values = []

                # Extend the values list with non-NaN values from the column
                values.extend(df[unique_col].values.tolist())

                # Flatten the nested list
                if len(values) != 1:
                    values = [item for sublist in values for item in sublist]

                # Add the values list as a column in the concatenated DataFrame
                concatenated_df[unique_col] = pd.Series(values)

            # Replace the original DataFrame with the concatenated one
            df = concatenated_df

            # Formal step of translating to the time-series panel data
            for unique_col in unique_columns:
                if is_cost:
                    df[['成本类别', unique_col, '年份']] = df[unique_col].str.split('!', expand=True)
                else:
                    df[['电厂分类', '电厂', '所在电网', '地理位置', unique_col, '年份']] = df[unique_col]. \
                        str.split('!', expand=True)

            # let cost category as the first column
            if is_cost:
                columns_to_move = '成本类别'
                cols = df.columns.tolist()
                cols.insert(0, cols.pop(cols.index(columns_to_move)))
                df = df[cols].sort_values(by=['年份']).reset_index(drop=True)
            # power sale
            else:
                columns_to_move = ['电厂分类', '电厂', '所在电网', '地理位置']
                # Reorder columns by moving the specified columns to the front
                new_columns = columns_to_move + [col for col in df.columns if col not in columns_to_move]
                # Update the DataFrame with the new column order
                df = df[new_columns].sort_values(by=['年份']).reset_index(drop=True)

            # Empty Values processing
            df.replace(['-', '/', '  -   ', ' – ', ''], np.nan, inplace=True)
            df.replace({r'\n': ''}, regex=True, inplace=True)

            # Update the corresponding class attribute - df
            setattr(self, i, df)

        self.preprocessed = True
        return self

    def translate_cells(self, local_dict_name=None, update=True):
        """
        Translates cells in the dataframe using a local dictionary or by performing translations.

        :param local_dict_name: The local dictionary file in JSON format, mapping Chinese to English
        :param update: If True, updates the local dictionary and performs translations as needed.
                       If False, uses the local_dict_name directly without updating. Defaults to True.
        :return: Returns the instance of the class with the translated cells.
        """
        # Load the local translation dict if the local_dict_name is provided
        translations_all = {}
        if local_dict_name is not None:
            try:
                with open(local_dict_name, "r") as f:
                    translations_all = json.load(f)
            except FileNotFoundError:
                pass

        # match digits
        regex = re.compile(r'^[-+]?\d{1,3}(?:,?\d{3})*(?:\.\d+)?$')

        for i in self.df_lst:
            df = getattr(self, i)
            translations = {}
            translator = Translator()
            # changed if the dictionary need to be updated
            if update:
                for column in df.columns:
                    if column not in translations_all:
                        # translate column
                        translated_text = translator.translate(column).text.strip()
                        if translated_text.lower() != column.lower():
                            translations[column] = translated_text

                    # Skip columns that contain only Digits values
                    first_valid_index = df[column].first_valid_index()
                    if first_valid_index is not None:
                        check_item = df[column].iloc[first_valid_index]
                        if (isinstance(check_item, str) and regex.match(check_item)) \
                                or isinstance(check_item, (int, float, np.number)):
                            continue

                    unique_elements = df[column].unique()
                    for element in unique_elements:
                        if element in translations_all:
                            continue
                        if regex.match(element):
                            continue
                        translated_text = translator.translate(element).text.strip()
                        if translated_text.lower() != element.lower():
                            translations[element] = translated_text
            # Update the local dictionary with the new translations
            translations_all.update(translations)

            # Update the DataFrame with translations
            df.replace(translations_all, inplace=True)
            df.rename(columns=translations_all, inplace=True)
            # Update the corresponding class attribute - df
            setattr(self, i, df)

        # Save the updated translations_all dictionary to the file
        if local_dict_name is not None:
            with open(local_dict_name, "w") as f:
                json.dump(translations_all, f)

        return self

    def correct_digital_type(self):
        name_pattern = r'^[A-Za-z\s]+$'

        for i in self.df_lst:
            df = eval('self.' + i)
            for col in df.columns:
                try:
                    # Skip columns that contain only string values
                    check_index = 0
                    check_item = df[col].iloc[check_index]
                    while pd.isna(df[col].iloc[check_index]):
                        check_index += 1
                        check_item = df[col].iloc[check_index]
                    if df[col].dtype == 'object' and re.match(name_pattern, check_item):
                        continue
                    if col == 'cost category':
                        continue
                    if is_numeric_dtype(df[col]):
                        continue
                    df[col] = df[col].str.replace(',', '')
                    # df[col] = df[col].apply(lambda x: ''.join(c for c in x if c.isdigit()))
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except ValueError:
                    continue
                except TypeError:
                    continue

        return self

    def scenario(self, types="power", pct=0.5, year=2020, coal_price=None,
                 coal_power=True, coal_purchase=False, coal_transportation=False):
        """
         Perform scenario tasks.

        :param types: (str) Type of scenario to generate. Default is "power".
        :param pct: (float) Percentage reduction in the scenario compared to the current state. Default is 0.5.
        :param year: (int) Year for which the scenario is generated. Default is 2020.
        :param coal_price: (float) Price of coal. Default is None from the 2020. #TODO (same as the year)
        :param coal_power: (bool) Indicates whether coal power is being used. Default is True.
        :param coal_purchase: (bool) Indicates whether coal is being purchased. Default is False.
        :param coal_transportation: (bool) Indicates whether coal is being transported. Default is False.
        :return: (pandas.DataFrame) Scenario case dataframe.
        """

        # if types == "power":
        # All Power
        year = str(year)
        electricity_sale_price = \
            self.power_sale['Total electricity sales (100 million kWh) ' + year] / \
            self.power_sale['Total electricity sales (100 million kWh) ' + year].sum() * \
            self.power_sale['Electricity Sales Price (RMB/MWh) ' + year]

        # Estimated Sales Cost
        sale_cost_lst = []
        sale_cost = self.power_cost['Cost (RMB million) ' + year].sum() - \
                    self.power_cost[self.power_cost['cost category'] == 'Maintenance fees'][
                        'Cost (RMB million) ' + year].item() - \
                    self.power_cost[self.power_cost['cost category'] == 'Other operating costs'][
                        'Cost (RMB million) ' + year].item() - \
                    self.power_cost[self.power_cost['cost category'] == 'Electricity cost'][
                        'Cost (RMB million) ' + year].item()

        sale_cost_lst.append(sale_cost)
        coal_power_sc = self.elec_cost['Cost (RMB million) ' + year].sum() - \
                        self.elec_cost[self.elec_cost['cost category'] == 'Maintenance fees'][
                            'Cost (RMB million) 2020'].item()
        sale_cost_lst.append(coal_power_sc)
        sale_cost_lst.append(sale_cost - coal_power_sc)
        sale_cost_lst.append(sale_cost - coal_power_sc)
        sale_cost_lst.append(0)

        group = self.power_sale.groupby('Power Plant Classification')[
            ['Total electricity sales (100 million kWh) ' + year, 'Electricity Sales Price (RMB/MWh) ' + year]]

        coal_types_total = group.apply(lambda df: (df.sum()))

        power_dispatch = coal_types_total['Total electricity sales (100 million kWh) ' + year].values.tolist()
        total_sales = coal_types_total['Total electricity sales (100 million kWh) ' + year].sum()
        power_dispatch.insert(0, total_sales)

        weighted_average_tariff = group.apply(lambda df: (df['Total electricity sales (100 million kWh) ' + year]
                                                          / df[
                                                              'Total electricity sales (100 million kWh) ' + year].sum()
                                                          * df[
                                                              'Electricity Sales Price (RMB/MWh) ' + year]).sum()
                                              ).values.tolist()
        weighted_average_tariff.insert(0, electricity_sale_price.sum())

        forecasted_gross_profit = group.apply(lambda df: (df['Total electricity sales (100 million kWh) ' + year] *
                                                          df['Electricity Sales Price (RMB/MWh) ' + year] / 10).sum()
                                              )
        forecasted_gross_profit_lst = forecasted_gross_profit.values.tolist()
        forecasted_gross_profit_lst.insert(0, forecasted_gross_profit.sum())

        lst1 = (np.array(forecasted_gross_profit_lst) - np.array(sale_cost_lst)).tolist()
        lst1[2] = forecasted_gross_profit_lst[2] + forecasted_gross_profit_lst[3] - sale_cost_lst[2]
        lst1[3] = lst1[2]

        # lst2 = ((np.array(forecasted_gross_profit_lst) - np.array(sale_cost_lst)) / (
        #     np.array(forecasted_gross_profit_lst))).tolist()
        forecasted_gross_profit_lst_copy = forecasted_gross_profit_lst.copy()  # create a copy of the original list
        last_two_sum = sum(forecasted_gross_profit_lst_copy[-3:-1])
        forecasted_gross_profit_lst_copy[-3:-1] = [last_two_sum, last_two_sum]
        lst2 = (np.array(lst1) / np.array(forecasted_gross_profit_lst_copy)).tolist()

        data = {
            'Power Dispatched (TWh)': power_dispatch,
            'Weighted Average Tariff (RMB/MWh)': weighted_average_tariff,
            'Estimated Gross Revenue (RMB/MWh)': forecasted_gross_profit_lst,
            'Estimated Sales Cost (Million RMB)': sale_cost_lst,
            'Gross Profit (Million RMB)': lst1,
            'Gross Profitability': lst2
        }
        self.data = data

        # case
        table = pd.DataFrame(data)
        table = table.drop(index=table.index[-1])
        power_sources = ['All Power', 'Coal Power', 'Gas Power', 'Hydro Power']
        table.set_index(pd.Index(power_sources), inplace=True)
        table['Power Dispatched (TWh)'] = table['Power Dispatched (TWh)'] / 10
        table['Gross Profitability'] = table['Gross Profitability'].apply(lambda x: '{:.2%}'.format(x))

        unit_cost = self.elec_cost.set_index('cost category')['Unit cost (yuan/MWh) in ' + year][
            'Raw materials, fuels and power']
        if coal_price is not None:
            if coal_price > table.loc['Coal Power', 'Weighted Average Tariff (RMB/MWh)']:
                year_higher = '2021'
                unit_cost = self.elec_cost.set_index('cost category')['Unit cost (yuan/MWh) in ' + year_higher][
                    'Raw materials, fuels and power']
            table['Weighted Average Tariff (RMB/MWh)'] = coal_price

        # Estimated Revenue& Cost under different coal reduction scenario using 2020 data
        value2 = table['Power Dispatched (TWh)']['Coal Power'] * (1 - pct) * \
                 table['Weighted Average Tariff (RMB/MWh)']['Coal Power']

        value3 = table['Power Dispatched (TWh)']['Coal Power'] * (1 - pct) * unit_cost + \
                 self.elec_cost.set_index('cost category')['Cost (RMB million) ' + year]['Labor cost'] + \
                 self.elec_cost.set_index('cost category')['Cost (RMB million) ' + year][
                     'Depreciation and amortization'] \
                 + self.elec_cost.set_index('cost category')['Cost (RMB million) ' + year]['other costs']
        case = {
            'Power Dispatched (TWh)': table['Power Dispatched (TWh)']['Coal Power'] * (1 - pct),
            'Weighted Average Tariff (RMB/MWh)': table['Weighted Average Tariff (RMB/MWh)']['Coal Power'],
            'Estimated Gross Revenue (RMB/MWh)': value2,
            'Estimated Sales Cost (Million RMB)': value3,
            'Gross Profit (Million RMB)': value2 - value3,
            'Gross Profitability': (value2 - value3) / value2
        }
        self.case = pd.DataFrame(case, index=['Reduced by ' + str(int(pct * 100)) + '%'])
        self.case['Gross Profitability'] = self.case['Gross Profitability'].apply(lambda x: '{:.2%}'.format(x))
        self.case = self.case.T

        return table

    def scenario2(self, types="power", pct=0.5, year=2020, coal_price=None,
                  coal_power=True, coal_purchase=False, coal_transportation=False):
        # Set index to 'years' if not already set
        if self.power_sale.index.name != 'years':
            self.power_sale.set_index('years', inplace=True)
        if self.power_cost.index.name != 'years':
            self.power_cost.set_index('years', inplace=True)
        if self.elec_cost.index.name != 'years':
            self.elec_cost.set_index('years', inplace=True)

        electricity_sale_price = \
            self.power_sale['Total electricity sales (100 million kWh)'][year] / \
            self.power_sale['Total electricity sales (100 million kWh)'][year].sum() * \
            self.power_sale['Electricity sales price (RMB/MWh)'][year]

        # Estimated Sales Cost
        sale_cost_lst = []
        sale_cost = self.power_cost['Cost (RMB million)'][year].sum() - \
                    self.power_cost[self.power_cost['cost category'] == 'Maintenance fees'][
                        'Cost (RMB million)'][year].item() - \
                    self.power_cost[self.power_cost['cost category'] == 'Other operating costs'][
                        'Cost (RMB million)'][year].item() - \
                    self.power_cost[self.power_cost['cost category'] == 'Electricity cost'][
                        'Cost (RMB million)'][year].item()

        sale_cost_lst.append(sale_cost)
        coal_power_sc = self.elec_cost['Cost (RMB million)'][year].sum() - \
                        self.elec_cost[self.elec_cost['cost category'] == 'Maintenance fees'][
                            'Cost (RMB million)'][year].item()
        sale_cost_lst.append(coal_power_sc)
        sale_cost_lst.append(sale_cost - coal_power_sc)
        sale_cost_lst.append(sale_cost - coal_power_sc)
        sale_cost_lst.append(0)

        group = self.power_sale.groupby(['years', 'Power Plant Classification'])
        # checks if the first element of the MultiIndex (i.e., the year) matches the specified year.
        group_filtered = group.filter(lambda x: x.name[0] == year)

        coal_types_total = group.sum(numeric_only=True).loc[year]

        power_dispatch = coal_types_total['Total electricity sales (100 million kWh)'].values.tolist()
        total_sales = coal_types_total['Total electricity sales (100 million kWh)'].sum()
        power_dispatch.insert(0, total_sales)

        weighted_average_tariff = group_filtered.groupby('Power Plant Classification').apply(
            lambda df: (
                    df['Total electricity sales (100 million kWh)']
                    / df['Total electricity sales (100 million kWh)'].sum()
                    * df['Electricity sales price (RMB/MWh)']
            ).sum()
        ).values.tolist()
        weighted_average_tariff.insert(0, electricity_sale_price.sum())

        # Estimated Gross Revenue
        forecasted_gross_profit = group_filtered.groupby('Power Plant Classification').apply(
            lambda df: (
                    df['Total electricity sales (100 million kWh)']
                    * df['Electricity sales price (RMB/MWh)'] / 10
            ).sum()
        ).values.tolist()
        forecasted_gross_profit.insert(0, sum(forecasted_gross_profit))

        lst1 = (np.array(forecasted_gross_profit) - np.array(sale_cost_lst)).tolist()
        lst1[2] = forecasted_gross_profit[2] + forecasted_gross_profit[3] - sale_cost_lst[2]
        lst1[3] = lst1[2]

        forecasted_gross_profit_lst_copy = forecasted_gross_profit.copy()  # create a copy of the original list
        last_two_sum = sum(forecasted_gross_profit_lst_copy[-3:-1])
        forecasted_gross_profit_lst_copy[-3:-1] = [last_two_sum, last_two_sum]
        lst2 = (np.array(lst1) / np.array(forecasted_gross_profit_lst_copy)).tolist()

        data = {
            'Power Dispatched (TWh)': power_dispatch,
            'Weighted Average Tariff (RMB/MWh)': weighted_average_tariff,
            'Estimated Gross Revenue (RMB/MWh)': forecasted_gross_profit,
            'Estimated Sales Cost (Million RMB)': sale_cost_lst,
            'Gross Profit (Million RMB)': lst1,
            'Gross Profitability': lst2
        }
        self.data = data

        # case
        table = pd.DataFrame(data)
        table = table.drop(index=table.index[-1])
        power_sources = ['All Power', 'Coal Power', 'Gas Power', 'Hydro Power']
        table.set_index(pd.Index(power_sources), inplace=True)
        table['Power Dispatched (TWh)'] = table['Power Dispatched (TWh)'] / 10
        table['Gross Profitability'] = table['Gross Profitability'].apply(lambda x: '{:.2%}'.format(x))

        unit_cost = self.elec_cost.set_index('cost category', append=True)['Unit cost (yuan/MWh)'][year][
            'Raw materials, fuels and power']

        if coal_price is not None:
            if coal_price > table.loc['Coal Power', 'Weighted Average Tariff (RMB/MWh)']:
                year_higher = year + 1
                unit_cost = self.elec_cost.set_index('cost category', append=True)['Unit cost (yuan/MWh)'][year_higher][
                    'Raw materials, fuels and power']
            table['Weighted Average Tariff (RMB/MWh)'] = coal_price

        # Estimated Revenue & Cost under different coal reduction scenario using specified year data
        value2 = table['Power Dispatched (TWh)']['Coal Power'] * (1 - pct) * \
                 table['Weighted Average Tariff (RMB/MWh)']['Coal Power']

        value3 = table['Power Dispatched (TWh)']['Coal Power'] * (1 - pct) * unit_cost + \
                 self.elec_cost.set_index('cost category', append=True)['Cost (RMB million)'][year]['Labor cost'] + \
                 self.elec_cost.set_index('cost category', append=True)['Cost (RMB million)'][year][
                     'Depreciation and amortization'] \
                 + self.elec_cost.set_index('cost category', append=True)['Cost (RMB million)'][year]['other costs']
        case = {
            'Power Dispatched (TWh)': table['Power Dispatched (TWh)']['Coal Power'] * (1 - pct),
            'Weighted Average Tariff (RMB/MWh)': table['Weighted Average Tariff (RMB/MWh)']['Coal Power'],
            'Estimated Gross Revenue (RMB/MWh)': value2,
            'Estimated Sales Cost (Million RMB)': value3,
            'Gross Profit (Million RMB)': value2 - value3,
            'Gross Profitability': (value2 - value3) / value2
        }
        self.case = pd.DataFrame(case, index=['Reduced by ' + str(int(pct * 100)) + '%'])
        self.case['Gross Profitability'] = self.case['Gross Profitability'].apply(lambda x: '{:.2%}'.format(x))
        self.case = self.case.T

        return table
