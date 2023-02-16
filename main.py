from Modeling import soeModel





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = soeModel()
    model.preprocessing()
    model.translate_col()
    model.translate_cells()
    print(model.power_cost.columns)