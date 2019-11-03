# GAbin
Continuous Variable Binning Algorithm to Maximize Information Value using Genetic Algorithm

## Download the paper: 
[Full paper download](https://link.springer.com/chapter/10.1007/978-3-030-32475-9_12)

## Cite the paper: 
Vejkanchana N., Kucharoen P. (2019) Continuous Variable Binning Algorithm to Maximize Information Value Using Genetic Algorithm. In: Florez H., Leon M., Diaz-Nafria J., Belli S. (eds) Applied Informatics. ICAI 2019. Communications in Computer and Information Science, vol 1051. Springer, Cham

## Setup with anaconda
```bash
conda create --name deap python=3.7
conda activate deap
pip install spyder
pip install deap
pip install tqdm
pip install numpy
pip install pandas
pip install matplotlib
```

## Run the example file
Change to output folder name at line 177 in GA_bin_all_para_FICO.py
In the example the folder name is FICO_woe_pic_all.

```bash
conda activate deap
python GA_bin_all_para_FICO.py
```
