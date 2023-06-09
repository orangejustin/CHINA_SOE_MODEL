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
    # TODO: table, matrix (variables) over time (time series table over the years)

    # TODO: Other SOEs.
    # TODO: updated CSVs in folder.

    def __init__(self):
        """
        upload raw data & initialization
        """
        # coal types
        self.preprocessed = False
        self.translated = False
        self.correct = False

        # by coal power
        power_sale = "raw data 2/Power Sale.csv"
        power_cost = "raw data 2/Power Costs.csv"
        elec_cost = "raw data 2/Electricity Costs.csv"
        # by coal types
        coal_oper = "raw data/Coal Operation Data.csv"
        coal_seg = "raw data/Coal segment operating costs.csv"
        coal_sale = "raw data/Coal sales price.csv"

        self.power_sale = pd.read_csv(power_sale, header=None)
        self.power_cost = pd.read_csv(power_cost, header=None)
        self.elec_cost = pd.read_csv(elec_cost, header=None)
        self.coal_oper = pd.read_csv(coal_oper, header=None)
        self.coal_seg = pd.read_csv(coal_seg, header=None)
        self.coal_sale = pd.read_csv(coal_sale, header=None)

        # self.df_lst = [self.power_sale, self.power_cost, self.elec_cost]
        self.df_lst = ['power_sale', 'power_cost', 'elec_cost', 'coal_oper', 'coal_seg', 'coal_sale']
        self.data = None
        self.case = None

    def preprocessing(self):
        """
        This method performs the second stage of preprocessing on the DataFrame.

        :return: Returns the instance of the class with the preprocessed DataFrame.
        """
        pd.set_option('mode.chained_assignment', None)

        if self.preprocessed:
            return self
        is_cost = False

        for i in self.df_lst:
            # if is cost type format
            if i == "power_cost" or i == "elec_cost" or i == "coal_sale" or i == "coal_seg" \
                    or i == "coal_oper":
                is_cost = True

            df = eval('self.' + i)
            if i == "power_sale":
                df.at[0, 0] = np.nan
                df = df[df[0] != '燃煤电厂合计/加权平均']
            if i == "coal_oper":
                row_to_insert = [np.nan, "(百万顿)", "(百万顿)"]
                # Create a DataFrame for the row to be inserted
                row_df = pd.DataFrame([row_to_insert], columns=df.columns, index=[1])
                # Insert the row into the DataFrame
                df = pd.concat([df.iloc[:1], row_df, df.iloc[1:]]).reset_index(drop=True)

            # column names reformatting
            df.loc[0] = df.loc[0].astype(str).str.extract(r'(\d{4})', expand=False).fillna(method='ffill')
            df = df.fillna('')
            df.iloc[1] = df.iloc[1].astype(str).str.replace('\n', ' ').str.strip()

            columns = df.iloc[1] + ' ' + df.iloc[0]

            # if the data is a cost-related table
            if is_cost:
                columns = columns.replace(' ', '成本类别')

            df.columns = columns

            # drop the unnecessary rows
            df = df.iloc[2:].reset_index(drop=True)
            df.columns = [str(col).strip() for col in df.columns]

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
                elif i == "power_sale":
                    df[col] = df['电厂分类'] + '!' + df['电厂'] + '!' + \
                              df['所在电网'] + '!' + df['地理位置'] + '!' + df[col]
                else:
                    return "Not available right now"

            # Drop and will be adding back afterward
            if is_cost:
                df.drop(['成本类别'], axis=1, inplace=True)
            else:
                df.drop(['电厂分类', '电厂', '所在电网', '地理位置'], axis=1, inplace=True)

            # Rename the col
            df.rename(columns=new_column_names, inplace=True)
            # Drop the unnecessary columns
            if not is_cost:
                df.drop(['预测毛利润 (百万元)', '计算I38中的加权平均售电电价', '计算Q38中的加权平均售电电价',
                         '总发电量 (亿千瓦时)', '平均利用小时 (小时)', '售电标准煤耗 (克/千瓦时)'], axis=1,
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

            if i == "coal_seg":
                df.rename(columns={"(百万元)": "cost"}, inplace=True)
            if i == "coal_sale":
                df.rename(columns={"成本类别": "sales"}, inplace=True)
            if i == "coal_oper":
                df.rename(columns={"成本类别": "煤炭产量"}, inplace=True)
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
                if col == "sales":
                    continue
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

    def fill_na(self):
        for i in :
            df = df.fillna(0)

    def scenario(self, types="coal power", pct=0.5, year=2020, coal_price=None,
                 coal_purchase=False, coal_transportation=False,
                 reduce_self_produced=False, reduce_purchased=False, cut_purchased_coal=False):
        """
        Perform scenario tasks.

        :param types: (str) Type of scenario to generate. Default is "coal power".
        :param pct: (float) Percentage reduction in the scenario compared to the current state. Default is 0.5.
        :param year: (int) Year for which the scenario is generated. Default is 2020.
        :param coal_price: (float) Price of coal. Default is None from the 2020.
        :param coal_purchase: (bool) Indicates whether coal is being purchased. Default is False.
        :param coal_transportation: (bool) Indicates whether coal is being transported. Default is False.
        :param reduce_purchased: TODO
        :param reduce_self_produced: TODO
        :param cut_purchased_coal: TODO
        :return: (pandas.DataFrame) Scenario case dataframe.
        """
        if types == "coal power":
            data = self.power(year)
            self.data = data
            # case
            table = self.power_case(data, year, coal_price, pct)
        elif types == "coal types":
            data = self.types(year)
            self.data = data
            table = self.types_case(data, coal_price, pct, reduce_self_produced,
                                    reduce_purchased, cut_purchased_coal)
            table = table
        else:
            return None

        return table

    def power(self, year):
        """
        This method calculates the various power and cost attributes for a given year.

        Parameters:
        year (int): The year for which the calculations need to be made.

        Returns:
        dict: A dictionary with calculated power attributes such as 'Power Dispatched (TWh)', 'Weighted Average Tariff (RMB/MWh)',
              'Estimated Gross Revenue (RMB/MWh)', 'Estimated Sales Cost (Million RMB)', 'Gross Profit (Million RMB)',
              and 'Gross Profitability'.
        """

        # constant variables
        included_wind = 4
        included_solar = 5
        custom_order = ['coal burning', 'gas', 'hydropower']

        # set index to 'years' if not already set
        if self.power_sale.index.name != 'years':
            self.power_sale.set_index('years', inplace=True)
        if self.power_cost.index.name != 'years':
            self.power_cost.set_index('years', inplace=True)
        if self.elec_cost.index.name != 'years':
            self.elec_cost.set_index('years', inplace=True)

        # calculate electricity_sale_price
        electricity_sale_price = \
            self.power_sale['Total electricity sales (100 million kWh)'][year] / \
            self.power_sale['Total electricity sales (100 million kWh)'][year].sum() * \
            self.power_sale['Electricity sales price (RMB/MWh)'][year]

        # Estimated Sales Cost
        sale_cost_lst = []

        # All Power
        sale_cost = self.power_cost['Cost (RMB million)'][year].sum() - \
                    self.power_cost[self.power_cost['cost category'] == 'Maintenance fees'][
                        'Cost (RMB million)'][year].item() - \
                    self.power_cost[self.power_cost['cost category'] == 'Other operating costs'][
                        'Cost (RMB million)'][year].item() - \
                    self.power_cost[self.power_cost['cost category'] == 'Electricity cost'][
                        'Cost (RMB million)'][year].item()

        sale_cost_lst.append(sale_cost)

        # Coal Power: all cost - Maintenance fees (维修费)
        coal_power_sc = self.elec_cost['Cost (RMB million)'][year].sum() - \
                        self.elec_cost[self.elec_cost['cost category'] == 'Maintenance fees'][
                            'Cost (RMB million)'][year].item()

        print(sale_cost)
        sale_cost_lst.append(coal_power_sc)

        # other variable cost: all sale cost - coal power cost
        other_cost = sale_cost - coal_power_sc

        group = self.power_sale.groupby(['years', 'Power Plant Classification'])

        # checks if the first element of the MultiIndex (i.e., the year) matches the specified year.
        group_filtered = group.filter(lambda x: x.name[0] == year)
        coal_types_total = group.sum(numeric_only=True).loc[year]

        # define the custom order
        if len(coal_types_total) == included_wind:
            custom_order = ['coal burning', 'gas', 'hydropower', 'wind energy']
        if len(coal_types_total) == included_solar:
            custom_order = ['coal burning', 'gas', 'hydropower', 'wind energy', 'Photovoltaic']

        # for additional all power needs plus one
        diff = len(coal_types_total) - len(sale_cost_lst) + 1
        sale_cost_lst = np.append(sale_cost_lst, [other_cost] * diff)

        # reindex based on custom order
        coal_types_total = coal_types_total.reindex(custom_order)

        # calculate power dispatch
        power_dispatch = coal_types_total['Total electricity sales (100 million kWh)'].values.tolist()
        total_sales = coal_types_total['Total electricity sales (100 million kWh)'].sum()
        power_dispatch.insert(0, total_sales)

        # calculate weighted_average_tariff
        weighted_average_tariff = group_filtered.groupby('Power Plant Classification').apply(
            lambda df: (
                    df['Total electricity sales (100 million kWh)']
                    / df['Total electricity sales (100 million kWh)'].sum()
                    * df['Electricity sales price (RMB/MWh)']
            ).sum()
        )

        # reorders the dataframe
        weighted_average_tariff = weighted_average_tariff.reindex(custom_order)
        weighted_average_tariff = weighted_average_tariff.values.tolist()

        # includes total electricity sale price
        weighted_average_tariff.insert(0, electricity_sale_price.sum())

        # Estimated Gross Revenue
        forecasted_gross_revenue = group_filtered.groupby('Power Plant Classification').apply(
            lambda df: (
                    df['Total electricity sales (100 million kWh)']
                    * df['Electricity sales price (RMB/MWh)'] / 10
            ).sum()
        )

        # reorder the dataframe
        forecasted_gross_revenue = forecasted_gross_revenue.reindex(custom_order)
        forecasted_gross_revenue = forecasted_gross_revenue.values.tolist()

        # include total forecasted_gross_revenue sale price
        forecasted_gross_revenue.insert(0, sum(forecasted_gross_revenue))

        gross_profit = (np.array(forecasted_gross_revenue) - np.array(sale_cost_lst)).tolist()
        other_sale_cost = forecasted_gross_revenue[2] + forecasted_gross_revenue[3] - sale_cost_lst[2]
        gross_profit[2:] = [other_sale_cost] * diff

        # defines as All Power, Coal Power and all other types
        forecasted_gross_revenue_copy = forecasted_gross_revenue.copy()  # create a copy of the original list
        other_revenue = sum(forecasted_gross_revenue[2:])
        forecasted_gross_revenue_copy[2:] = [other_revenue] * diff

        # calculates gross_profitability
        with np.errstate(divide='ignore', invalid='ignore'):
            gross_profitability = (np.array(gross_profit) / np.array(forecasted_gross_revenue_copy))

        # Change any NaNs or Infs (which could occur from division by zero) to 0
        gross_profitability[~np.isfinite(gross_profitability)] = 0
        gross_profitability = gross_profitability.tolist()

        data = {
            'Power Dispatched (TWh)': power_dispatch,
            'Weighted Average Tariff (RMB/MWh)': weighted_average_tariff,
            'Estimated Gross Revenue (RMB/MWh)': forecasted_gross_revenue,
            'Estimated Sales Cost (Million RMB)': sale_cost_lst,
            'Gross Profit (Million RMB)': gross_profit,
            'Gross Profitability': gross_profitability
        }

        return data

    def power_case(self, data, year, coal_price, pct):
        """
        This function is used to generate a report of a power case in a given year.

        :param data: The input data which is expected to be a DataFrame including 'Power Dispatched (TWh)', 'Weighted Average Tariff (RMB/MWh)',
                     'Estimated Gross Revenue (RMB/MWh)', 'Estimated Sales Cost (Million RMB)', and 'Gross Profitability' columns.
        :param year: The year when the power case takes place. It is used to align with data of unit cost and electric cost.
        :param coal_price: The coal price for the power case. If it's provided and higher than the current Weighted Average Tariff for Coal Power,
                           the function will update the unit cost and Weighted Average Tariff in the table.
        :param pct: The percentage of the power that is reduced in the case.

        :return: The updated DataFrame which includes the power case situation.

        The function does the following:

        - Checks the number of power sources and defines them accordingly.
        - Sets rows with 'Estimated Gross Revenue (RMB/MWh)' of 0 to all 0.
        - Adjusts the unit cost if a coal_price is provided and it's higher than the current Weighted Average Tariff for Coal Power.
        - Calculates the Estimated Revenue & Cost under different coal reduction scenarios using the specified year data.
        - Adds the scenario into a case and formats the 'Gross Profitability' as a percentage.
        """
        table = pd.DataFrame(data)
        included_wind = 5
        included_solar = 6

        # set entire row to 0 if 'Estimated Gross Revenue (RMB/MWh)' is 0
        table.loc[table['Estimated Gross Revenue (RMB/MWh)'] == 0] = 0

        if table.shape[0] == included_wind:
            power_sources = ['All Power', 'Coal Power', 'Gas Power', 'Hydro Power', 'wind']
        elif table.shape[0] == included_solar:
            power_sources = ['All Power', 'Coal Power', 'Gas Power', 'Hydro Power', 'wind', 'Photovoltaic']
        else:
            power_sources = ['All Power', 'Coal Power', 'Gas Power', 'Hydro Power']


        table.set_index(pd.Index(power_sources), inplace=True)
        table['Power Dispatched (TWh)'] = table['Power Dispatched (TWh)'] / 10
        table['Gross Profitability'] = table['Gross Profitability'].apply(lambda x: '{:.2%}'.format(x))

        unit_cost = self.elec_cost.set_index('cost category', append=True)['Unit cost (yuan/MWh)'][year][
            'Raw materials, fuels and power']

        # if we set up a coal price
        if coal_price is not None:
            # if that coal_price is higher than Weighted Average
            if coal_price > table.loc['Coal Power', 'Weighted Average Tariff (RMB/MWh)']:
                year_higher = year + 1
                unit_cost = self.elec_cost.set_index('cost category', append=True)['Unit cost (yuan/MWh)'] \
                    [year_higher]['Raw materials, fuels and power']
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

    def types(self, year):
        """
        Gross Revenue = Sales Volume * Average Coal Price
        Gross Profit = Gross Revenue - Sales Cost
        Sales Cost = Purchased Coal Sales Cost + Self-produced Coal Sales Cost
        Purchased Coal Sales Cost = Purchase Volumn * (Unit Purchase Cost + Unit Transportation Cost)
        Self-produced Coal Sales Cost = Fixed Cost + Self-produce Volume *
                                                        (Unit Material Cost + Unit Transportation Cost)
        Fixed cost = personnel + depreciations + other production cost

        :param year: the year we want to explore the analysis on
        :return: a dictionary contains sales_volume, average_coal_price, gross_revenue,
                                       estimated_sales_cost, gross_profit, and gross_profitability
        """
        # Set index to 'years' if not already set
        if self.coal_oper.index.name != 'years':
            self.coal_oper.set_index('years', inplace=True)
        if self.coal_seg.index.name != 'years':
            self.coal_seg.set_index('years', inplace=True)
        if self.coal_sale.index.name != 'years':
            self.coal_sale.set_index('years', inplace=True)

        data = {
            'sales_volume': [],
            'average_coal_price': [],
            'gross_revenue': [],
            'estimated_sales_cost': [],
            'gross_profit': [],
            'gross_profitability': []
        }

        # 运输(单位成本) = 运输成本 / 煤炭销售量
        transportation_unit = self.coal_seg[self.coal_seg['cost category'] == 'transportation cost']['cost'][year] / \
                              self.coal_oper[self.coal_oper['coal production'] == 'coal sales'][
                                  'Metrics (Millions of tons)'][year]
        self.transportation_unit = transportation_unit

        # 燃料(单位成本) = 原材料、燃料及动力 / 自产煤
        fuel_unit = self.coal_seg[self.coal_seg['cost category'] == 'Raw materials, fuels and power']['cost'][year] / \
                    self.coal_oper[self.coal_oper['coal production'] == 'self-produced coal'][
                        'Metrics (Millions of tons)'][year]
        self.fuel_unit = fuel_unit

        # 外购煤(单位采购成本) = 外购煤成本 / 外购煤
        purchased_coal_unit = self.coal_seg[self.coal_seg['cost category'] == 'Purchased coal cost']['cost'][year] / \
                              self.coal_oper[self.coal_oper['coal production'] == 'purchased coal'][
                                  'Metrics (Millions of tons)'][year]
        self.purchased_coal_unit = purchased_coal_unit

        # 自产煤固定成本 = 人工成本 + 折旧及摊销 + 其他成本
        fixed_cost_of_self_produced_coal = 0
        for i in ['Labor cost', 'Depreciation and amortization', 'other costs']:
            fixed_cost_of_self_produced_coal += self.coal_seg[self.coal_seg['cost category'] == i]['cost'][year]
            self.fixed_cost_of_self_produced_coal = fixed_cost_of_self_produced_coal

        # replicate table: 2020's estimated sales revenue & cost by coal types
        for i in ['coal sales', 'self-produced coal', 'purchased coal']:
            data['sales_volume'].append(
                self.coal_oper[self.coal_oper['coal production'] == i]['Metrics (Millions of tons)'][year])
            data['average_coal_price'].append(self.coal_sale[self.coal_sale['sales'] == 'i. Domestic sales']
                                              ['Price (excluding tax) (yuan/ton)'][year])
            if i == 'self-produced coal':
                temp_value = data['sales_volume'][1] * (
                        transportation_unit + fuel_unit) + fixed_cost_of_self_produced_coal
                data['estimated_sales_cost'].append(temp_value)

            if i == 'purchased coal':
                temp_value = data['sales_volume'][2] * (transportation_unit + purchased_coal_unit)
                data['estimated_sales_cost'].append(temp_value)

        data['gross_revenue'] = (np.array(data['sales_volume']) * np.array(data['average_coal_price'])).tolist()
        data['estimated_sales_cost'].insert(0, sum(data['estimated_sales_cost']))
        data['gross_profit'] = (np.array(data['gross_revenue']) - np.array(data['estimated_sales_cost'])).tolist()
        data['gross_profitability'] = (np.array(data['gross_profit']) / np.array(data['gross_revenue'])).tolist()

        return data

    def types_case(self, data, coal_price, pct, reduce_self_produced, reduce_purchased, cut_purchased_coal):
        # initialize columns values
        self.case = {
            'self_produced_coal_volume': None,
            'purchased_coal_volume': None,
            'total_coal_volume': None,
            'average_coal_price': None,
            'sales_revenue': None,
            'sales_cost': None,
            'gross_sales_profit': None,
            'gross_profitability': None
        }

        # Insert a space before all caps, then remove the leading space on the first word
        new_data = {re.sub(r"(?<=\w)([A-Z])", r" \1", key).title().replace('_', ' '): value
                    for key, value in data.items()}
        # replicate 2020's estimated sales revenue & cost by coal types
        table = pd.DataFrame(new_data)
        coal_sources = ['coal sales', 'self-produced coal', 'purchased coal']
        table.set_index(pd.Index(coal_sources), inplace=True)
        table['Gross Profitability'] = table['Gross Profitability'].apply(lambda x: '{:.2%}'.format(x))

        # Cases: Reduced Self-produced/Purchased/or both Coal by 50%
        self.reduce_coal_volumes(table, pct, reduce_self_produced, reduce_purchased, cut_purchased_coal)

        self.case['total_coal_volume'] = self.case['self_produced_coal_volume'] + self.case['purchased_coal_volume']
        if coal_price is None:
            self.case['average_coal_price'] = table['Average Coal Price']['coal sales']
        else:
            self.case['average_coal_price'] = coal_price

        self.case['sales_revenue'] = self.case['average_coal_price'] * self.case['total_coal_volume']
        self.case['sales_cost'] = self.fixed_cost_of_self_produced_coal \
                                  + self.case['total_coal_volume'] * self.transportation_unit \
                                  + self.purchased_coal_unit * self.case['purchased_coal_volume'] \
                                  + self.fuel_unit * self.case['self_produced_coal_volume']
        self.case['gross_sales_profit'] = self.case['sales_revenue'] - self.case['sales_cost']
        self.case['gross_profitability'] = self.case['gross_sales_profit'] / self.case['sales_revenue']
        self.case['gross_profitability'] = '{:.2%}'.format(self.case['gross_profitability'])
        self.case = {re.sub(r"(?<=\w)([A-Z])", r" \1", key).title().replace('_', ' '): value
                     for key, value in self.case.items()}
        self.case = pd.DataFrame(self.case, index=['Reduced by ' + str(int(pct * 100)) + '%'])
        self.case = self.case.T
        return table

    def reduce_coal_volumes(self, table, pct, reduce_self_produced=False, reduce_purchased=False,
                            cut_purchased_coal=False):
        # unchanged of purchased_coal_volume
        self.case['purchased_coal_volume'] = table['Sales Volume']['purchased coal']
        # unchanged of self_produced_coal_volume
        self.case['self_produced_coal_volume'] = table['Sales Volume']['self-produced coal']
        if cut_purchased_coal:
            result = table['Sales Volume']['purchased coal'] - 0.5 * table['Sales Volume']['coal sales']
            self.case['purchased_coal_volume'] = max(result, 0)
            if self.case['purchased_coal_volume'] == 0:
                self.case['self_produced_coal_volume'] = table['Sales Volume']['self-produced coal'] - \
                                                         (0.5 * table['Sales Volume']['coal sales'] -
                                                          table['Sales Volume']['purchased coal'])
            else:
                self.case['self_produced_coal_volume'] = table['Sales Volume']['self-produced coal']
            return self
        if reduce_self_produced:
            self.case['self_produced_coal_volume'] = table['Sales Volume']['self-produced coal'] * pct
        if reduce_purchased:
            self.case['purchased_coal_volume'] = table['Sales Volume']['purchased coal'] * pct

        return self
