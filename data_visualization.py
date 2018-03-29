# %%

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import wb

eu_countries = ['BE', 'BG', 'CZ', 'DK', 'DE', 'EE', 'IE', 'GR', 'ES', 'FR', 'HR',
                'IT', 'CY', 'LV', 'LT', 'LU', 'HU', 'MT', 'NL', 'AT', 'PL', 'PT',
                'RO', 'SI', 'SK', 'FI', 'SE', 'GB']

ue = wb.download(indicator="SL.UEM.TOTL.ZS",
                 country=eu_countries, start=1991,
                 end=2014)

ue.reset_index(inplace=True)

# ue.columns = ['country', 'year', 'unemployment']

#%%
ue_wide = ue.pivot(index='country', columns='year',
                   values='unemployment')

sns.heatmap(ue_wide)
sns.palplot(sns.color_palette('Greens_r', 7))

# %%
colors = ['#F5A422', '#3E22F5', '#3BF522',
          '#C722F5', '#F53E22']

pal = sns.color_palette(colors)

sns.palplot(pal)

# %%
mx = pd.read_csv('http://personal.tcu.edu/kylewalker/mexico.csv')
sns.barplot(x='gdp08', y='name',
            data=mx.sort_values('gdp08', ascending=False),
            palette="Greens_r")


# %%
sns.set_style('white')

ue['year2'] = ue.year.astype(float)

full = ue.pivot(index='year2', columns='country', values='unemployment')

greece = full['Greece']

full.plot(legend=False, style='lightgrey')
greece.plot(style='blue', legend=True)

plt.annotate('Global recession \nspreads to Europe', xy=(2009, 9.5),
             xycoords='data', xytext=(2005, 15), textcoords='data',
             arrowprops=dict(arrowstyle='simple', color='#000000'))


# %%
full.plot(subplots=True, layout=(7, 4),
          figsize=(22, 20), sharey=True)

# %%
ue_wide.sort_values(by=['2014', '2013', '2012', '2011', '2010'], ascending=False)
plt.figure(figsize=(10, 7))

sns.heatmap(ue_wide, cmap='YlGnBu')

plt.ylabel("")
plt.xlabel("")
plt.title("Unemployment in Europe, 1991-2013")
plt.xticks(rotation=45)

#%%
print(dir(ue_wide))


