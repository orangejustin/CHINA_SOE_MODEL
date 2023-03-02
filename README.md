# SOE Model
The `soeModel` package is a Python implementation of the State-Owned Enterprise (SOE) model for the Chinese electricity sector. The package includes a class for generating scenarios for the SOE model and analyzing the results.

### Requirements
- Python 3.x
- Pandas
- NumPy
- googletrans
- re
- json


### Installation
To use this package, you can simply clone the repository from GitHub:

```bash
git clone https://github.com/orangejustin/CHINA_SOE_MODEL.git
```
### Usage
To use the `soeModel` package, you need to import the soeModel class from the `Modeling` module

#### Importing the Class & Initializing the Class
```python
from Modeling import soeModel

model = soeModel()
```

#### Preprocessing
The `preprocessing()` function in the `soeModel` class preprocesses the data for the model. This function will automatically read the data from the `Data.csv` file in the repository and preprocess it.


```python
model.preprocessing()
```
#### Translating Chinese Characters

The `translate_cells()` function in the soeModel class translates Chinese characters in the data to English using a local dictionary. This function requires a 
dictionary file of Chinese characters and their English translations.
```python
model.translate_cells(local_dict_name="local_dictionary.json", local=True)
```
Note that you need to provide the name of the `local` dictionary file as well as set the local flag to `True` if the dictionary is in your local directory.

#### Correcting Digital Types

The `correct_digital_type()` function in the `soeModel` class corrects the types of digital data to float or int. This function will automatically detect columns with digital data and convert them to the correct type.


```python
model.correct_digital_type()
```


#### Running a Scenario
The `scenario()` function in the `soeModel` class
runs a scenario with the specified parameters. It will return 
a dictionary of the model's output.

```python
params = {
    "types": "power",
    "pct": 0.5,
    "year": 2020,
    "coal_price": 50.0,
    "coal_power": True,
    "coal_purchase": False,
    "coal_transportation": False
}

table = model.scenario(**params)
```
Finally, you can retrieve the results of the scenario by accessing the case attribute:

```python
print(model.case)
```

### License
This project is licensed under the MIT License. See the LICENSE file for more information.

### Acknowledgements
I would like to acknowledge the work of M. Davidson and colleagues from https://mdavidson.org/, whose research on modeling China's state-owned enterprise sector inspired this project. Specifically, I based my modeling code on their paper "China's State-Owned Enterprise Sector: Model-Based Analysis of Industrial Policy and Energy Consumption" (available at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4124504).