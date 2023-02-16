import pandas as pd
from googletrans import Translator
from pandas.api.types import is_numeric_dtype
import numpy as np
import re


class soeModel:
    def __init__(self):
        """
        upload raw data
        """
        # coal types
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
        preprocessing of the df
        :param df:
        :return:
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

    def translate_col(self):
        for i in self.df_lst:
            df = eval('self.' + i)
            translator = Translator()
            translate_list = []
            for col in df.columns:
                translate_list.append(translator.translate(col).text)
            # update
            df.columns = translate_list

        return self

    def translate_cells(self):
        for i in self.df_lst:
            df = eval('self.' + i)
            translations = {}
            translator = Translator()
            for column in df.columns:
                # unique elements of the column
                # exclude numeric numbers
                if '20' in column:
                    continue
                unique_elements = df[column].unique()
                for element in unique_elements:
                    # add translation to the dictionary
                    translations[element] = translator.translate(element).text.strip()

            # update
            df.replace(translations, inplace=True)

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
                    print(col)
                    df[col] = df[col].str.replace(',', '')
                    # df[col] = df[col].apply(lambda x: ''.join(c for c in x if c.isdigit()))
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except ValueError:
                    continue
                except TypeError:
                    continue

        return self

    def scenario(self, types):
        """
        scenario cases function in the working paper

        :param types: select a type
        :return: scenario case results
        """
        if types == "power":
            # All Power
            electricity_sale_price = \
                self.power_sale['Total electricity sales (100 million kWh) 2020'] / \
                self.power_sale['Total electricity sales (100 million kWh) 2020'].sum() * \
                self.power_sale['Electricity Sales Price (RMB/MWh) 2020']

            # Estimated Sales Cost
            sale_cost_lst = []
            sale_cost = self.power_cost['Cost (RMB million) 2020'].sum() - \
                        self.power_cost[self.power_cost['cost category'] == 'Maintenance fees'][
                            'Cost (RMB million) 2020'].item() - \
                        self.power_cost[self.power_cost['cost category'] == 'Other operating costs'][
                            'Cost (RMB million) 2020'].item() - \
                        self.power_cost[self.power_cost['cost category'] == 'Electricity cost'][
                            'Cost (RMB million) 2020'].item()

            sale_cost_lst.append(sale_cost)
            coal_power_sc = self.elec_cost['Cost (RMB million) 2020'].sum() - \
                            self.elec_cost[self.elec_cost['cost category'] == 'Maintenance fees'][
                                'Cost (RMB million) 2020'].item()
            sale_cost_lst.append(coal_power_sc)
            sale_cost_lst.append(sale_cost - coal_power_sc)
            sale_cost_lst.append(sale_cost - coal_power_sc)


            group = self.power_sale.groupby('Power Plant Classification')[
                ['Total electricity sales (100 million kWh) 2020', 'Electricity Sales Price (RMB/MWh) 2020']]

            coal_types_total = group.apply(lambda df: (df.sum()))

            power_dispatch = coal_types_total['Total electricity sales (100 million kWh) 2020'].values.tolist()
            total_sales = coal_types_total['Total electricity sales (100 million kWh) 2020'].sum()
            power_dispatch.insert(0, total_sales)

            weighted_average_tariff = group.apply(lambda df: (df['Total electricity sales (100 million kWh) 2020']
                                                              / df[
                                                                  'Total electricity sales (100 million kWh) 2020'].sum()
                                                              * df[
                                                                  'Electricity Sales Price (RMB/MWh) 2020']).sum()
                                                  ).values.tolist()
            weighted_average_tariff.insert(0, electricity_sale_price.sum())

            forecasted_gross_profit = group.apply(lambda df: (df['Total electricity sales (100 million kWh) 2020'] *
                                                              df['Electricity Sales Price (RMB/MWh) 2020'] / 10).sum()
                                                  )
            forecasted_gross_profit_lst = forecasted_gross_profit.values.tolist()
            forecasted_gross_profit_lst.insert(0, forecasted_gross_profit.sum())

            lst1 = (np.array(forecasted_gross_profit_lst[:-1]) - np.array(sale_cost_lst)).tolist()
            lst1.append(0)
            lst2 = ((np.array(forecasted_gross_profit_lst[:-1]) - np.array(sale_cost_lst)) / (
                np.array(forecasted_gross_profit_lst[:-1]))).tolist()

            lst2.append(0)

            sale_cost_lst.append(0)

        data = {
            'Power Dispatched (TWh)': power_dispatch,
            'Weighted Average Tariff (RMB/MWh)': weighted_average_tariff,
            'Estimated Gross Revenue (RMB/MWh)': forecasted_gross_profit_lst,
            'Estimated Sales Cost (Million RMB)': sale_cost_lst,
            'Gross Profit (Million RMB)': lst1,
            'Gross Profitability': lst2
        }
        self.data = data

        return pd.DataFrame(data)
