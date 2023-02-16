import pandas as pd
import numpy as np


class soeModel:

    def __init__(self):
        self.COAL_SALE_PRICE = 410
        cost = pd.read_excel(r'shenhua.xlsx', sheet_name='Coal Cost')
        cost = cost.replace('-', None)
        cost = cost.set_index('year')
        self.cost = cost
        operation = pd.read_excel(r'shenhua.xlsx', sheet_name='Operation Data')
        operation = operation.replace('-', None)
        operation = operation.set_index('year')
        self.operation = operation
        self.FC_SP_COAL = (cost['原材料、燃料及动力'] + cost['人工成本'] + cost['折旧及摊销'] + cost['其他生产成本'])[2020]

    def Sales_Revenue(self, year, type):
        if type == "Coal sales":
            gross_profit = self.operation['Coal sales'][year] * self.COAL_SALE_PRICE - self.cost['营业成本合计'][year]
            gross_profitability = gross_profit / (self.operation['Coal sales'][2020] * self.COAL_SALE_PRICE)

        elif type == "Self-produced coal":
            coal_cost = self.FC_SP_COAL + self.cost['运输成本'][2020] * (self.operation['Self-produced Coal'][2020]
                                                               / self.operation['Coal sales'][2020])

            gross_profit = self.operation['Self-produced Coal'][2020] * self.COAL_SALE_PRICE - coal_cost

            gross_profitability = gross_profit / (self.operation['Self-produced Coal'][2020] * self.COAL_SALE_PRICE)

        elif type == "Purchased coal":
            coal_cost = self.FC_SP_COAL + self.cost['运输成本'][2020] * (self.operation['Purchased Coal'][2020]
                                                               / self.operation['Coal sales'][2020])
            gross_profit = self.operation['Purchased Coal'][2020] * self.COAL_SALE_PRICE - coal_cost

            gross_profitability = gross_profit / (self.operation['Purchased Coal'][2020] * self.COAL_SALE_PRICE)

        else:
            gross_profit, gross_profitability = None, None

        print("gross_profit is: ", gross_profit)
        print("gross_profitability is:", gross_profitability)
        print()
        return gross_profit, gross_profitability

    def translate_col(self):
        return




    
    

