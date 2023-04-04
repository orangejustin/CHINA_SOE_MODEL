import pandas as pd
from googletrans import Translator
from pandas.api.types import is_numeric_dtype
import numpy as np
import re
import json


class soeModel:
    # TODO: parameters name
    # TODO: each table correspond each graph in the paper: needs index
    # TODO: Model table refine
    # TODO: All table updated
    # TODO: Debugs for the error

    def __init__(self):
        """
        upload raw data
        """
        # coal types
        self.case = None
        self.data = None
        power_sale = "raw data/Power Sale.csv"
        power_cost = "raw data/Power Costs.csv"
        elec_cost = "raw data/Electricity Costs.csv"

        self.power_sale = pd.read_csv(power_sale, header=None)
        self.power_cost = pd.read_csv(power_cost, header=None)
        self.elec_cost = pd.read_csv(elec_cost, header=None)

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

    def preprocessing_2(self, df):
        df = df.T.reset_index(drop=True)
        df.columns = df.loc[0].fillna(method='ffill').fillna('year')
        df.drop(index=[0], inplace=True)
        df['year'] = df['year'].astype(str)
        # apply regex and remove '\n(百万元)' from the 'year' column
        df['year'] = df['year'].str.replace('\n(百万元)', '', regex=True)
        # extract year digits using regex
        df['year'] = df['year'].apply(lambda s: re.findall('\d{4}', s)[0])
        df['year'] = df['year'].astype(int)
        return df

    def translate_cells(self, local_dict_name, local):
        # store a local translation dict
        try:
            with open(local_dict_name, "r") as f:
                translations_all = json.load(f)
        except FileNotFoundError:
            translations_all = {}

        # derive from local
        if local:
            for i in self.df_lst:
                df = eval('self.' + i)
                # update
                df.replace(translations_all, inplace=True)
                df.rename(columns=translations_all, inplace=True)
            return self

        regex = re.compile(r'^[-+]?\d{1,3}(?:,?\d{3})*(?:\.\d+)?$')

        # else perform translate
        for i in self.df_lst:
            df = eval('self.' + i)
            translations = {}
            translator = Translator()
            for column in df.columns:
                if column not in translations_all:
                    translations[column] = translator.translate(column).text.strip()
                # unique elements of the column
                # exclude numeric numbers
                if '20' in column:
                    continue
                unique_elements = df[column].unique()
                for element in unique_elements:
                    if element in translations_all:
                        continue
                    # if digit types
                    if regex.match(element):
                        continue
                    # add translation to the dictionary
                    translations[element] = translator.translate(element).text.strip()

            # update
            df.replace(translations, inplace=True)
            df.rename(columns=translations, inplace=True)
            # df.columns = translate_list
            translations_all.update(translations)

        try:
            with open(local_dict_name, "a") as f:
                json.dump(translations_all, f)
                f.write("\n")
        except FileNotFoundError:
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

        lst2 = (np.array(lst1) / np.array(sale_cost_lst)).tolist()

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
            'Gross Profitability': (value2 - value3) / value3
        }
        self.case = pd.DataFrame(case, index=['Reduced by ' + str(int(pct * 100)) + '%'])
        self.case['Gross Profitability'] = self.case['Gross Profitability'].apply(lambda x: '{:.2%}'.format(x))
        self.case = self.case.T

        return table
