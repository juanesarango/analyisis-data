# %%
from os.path import join
import os

from pandas_datareader import wb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = os.getcwd()
OUTPUT = join(ROOT, 'data')
INPUT = join(ROOT, 'fixed_stats')

PIPELINES = [
    (1, 'BATTENBERG'),
    (2, 'BRASS'),
    (3, 'CAVEMAN'),
    (4, 'CNVKIT'),
    (5, 'FLT3ITD'),
    (6, 'PINDEL'),
    (7, 'RNACALLER'),
    (8, 'MERGE_TABLES'),
    (9, 'ANNOT_CAVEMAN'),
    (10, 'ANNOT_PINDEL'),
    (11, 'ANNOT_CAVEMAN'),
    (12, 'ANNOT_PINDEL'),
    (13, 'PTD'),
    (14, 'QC_DATA'),
    (15, 'FACETS'),
    (16, 'RNAFUSIONS'),
    (17, 'CONPAIR')
]

# %% LOAD DATA IN DATAFRAME FROM JOBS FILES

def store_data(data, filename):
    data.to_csv(join(OUTPUT, f'{filename}.csv'))
    with open(join(OUTPUT, f'{filename}.txt'), 'w') as fo:
        fo.write(data.to_string())


def get_step_name(log_file):
    file_name = log_file.split('/')[-1]
    job_name = file_name.split('.')[0]
    step_name = job_name.split('_')
    return step_name[1] if len(step_name) > 1 else None


def aggregate_jobs():
    column_names = [
        "Analysis pk",
        "Technique",
        "BAM Size kb",
        "Pipeline",
        "LogFile",
        "Total Requested Memory",
        "Delta Memory",
        "Max Swap",
        "Run Time",
        "Step Name",
    ]
    frames = []
    for pipeline_id, _ in PIPELINES:
        stats_log = join(INPUT, f'stats_{pipeline_id}.txt')

        if not os.path.isfile(stats_log):
            continue

        df = pd.read_csv(
            stats_log,
            sep="\t",
            header=None,
            names=column_names
        )

        if not df.empty:
            df['Step Name'] = df.apply(
                lambda row: get_step_name(row["LogFile"]), axis=1
            )
            frames.append(df)

    # Concatenate all pipelines dataframes
    data = pd.concat(frames)

    # Write to files
    store_data(data, 'JOBS')
    return data


print('Creating Dataframes of Jobs')
JOBS = aggregate_jobs()
print(f'Finished! Collected {len(JOBS)} records.')

# %% GET % OF COMPLETED JOBS BY TIME THRESHOLD
import numpy as np

def format_count(count):
    count = str(count)
    return ''.join(
        ['  ' for _ in range(len(count), 10)]
    ) + count


def get_job_completeness(data, filters, thresholds):

    var = "Run Time"
    data[var] = data[var] / (60*60)

    for column, value in filters.items():
        data = data[data[column] == value]

    if data.empty:
        return None, 0

    completeness = {
        threshold: round(
            len(data[data[var] <= float(threshold)]) / len(data[var]) * 100, 1)
        for threshold in thresholds
    }

    distribution = data.agg({
        'Run Time': [
            'mean',
            'std',
            'min',
            'max'
        ],
    }).to_dict()['Run Time']

    # distribution = data.groupby("Step Name").agg({
    #     "Run Time": lambda x: np.percentile(x, 99)
    # }).to_dict()['Run Time'][filters["Step Name"]]

    return completeness, distribution, len(data)

def plot_pipelines(data, thresholds):

    stats = pd.DataFrame(
        columns=(
            ["Pipeline", "Technique", "Step Name", "Count"]
        )
    )

    pipelines = data["Pipeline"].unique()
    for pipeline in pipelines[:]:

        techniques = data[
            (data["Pipeline"] == pipeline)
        ]["Technique"].unique()
        for technique in techniques:

            steps = data[
                (data["Technique"] == technique) &
                (data["Pipeline"] == pipeline)
            ]["Step Name"].unique()

            for step in steps:

                if not step: continue

                step_stats = {
                    "Pipeline": pipeline,
                    "Technique": technique,
                    "Step Name": step,
                }

                completeness, distribution, count = get_job_completeness(
                    data.copy(),
                    step_stats,
                    thresholds
                )
                step_stats = {**step_stats, **completeness}
                step_stats["Count"] = count
                step_stats["2-std"] = distribution['mean'] + \
                    2 * distribution['std']
                # step_stats["2-std"] = distribution

                print(f'Done {technique}-{pipeline}-{step}')

                stats = stats.append(step_stats, ignore_index=True)

    store_data(stats, 'COMPLETENESS')
    return stats


THRESHOLDS = [
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1,
    1.5,
    2,
    2.5,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    20,
    50,
    100,
    200,
    300
]

JOBS_COPY = JOBS.copy()
STATS = plot_pipelines(JOBS_COPY[:], THRESHOLDS)
print(STATS)

# %% FORMATTING STATS
def complete_jobs(data, threshold):
    above_th = data[data[threshold] == 100]['Count'].sum()
    total = data['Count'].sum()
    percentage = abs(100 * above_th / total)
    # print(f'{threshold} -> {above_th} / {total} ')
    return f'{round(percentage, 2)} % - {threshold} hours'


def plot_heatmap(data):

    # Format Dataframes to plot
    # rev_thresholds = [i for i in reversed(THRESHOLDS)]
    # stats_sorted = data.sort_values(by=rev_thresholds)
    stats_sorted = data.sort_values(by=['2-std'], ascending=False)
    stats_sorted['Format Count'] = stats_sorted.apply(
        lambda row: format_count(row["Count"]), axis=1
    )
    stats_index = stats_sorted.set_index(
        ["Pipeline", "Technique", "Step Name", "Format Count"]
    )
    stats_index = stats_index.drop(["Count", "2-std"], axis=1)
    stats_index.index = [
        stats_index.index.map('{0[0]} {0[1]} {0[2]} {0[3]}'.format)
    ]
    stats_index.index.names = ['JOBS']

    # Plot Heatmap
    from matplotlib.colors import LogNorm

    plt.subplots(figsize=(6, 25))

    axes = sns.heatmap(
        stats_index.replace([0], 0.01),
        cbar_kws={"orientation": "horizontal"},
        # norm=LogNorm(vmin=100, vmax=0.01)
        cmap=sns.color_palette("GnBu_r"),
        # cmap=sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True),
        # cmap=sns.color_palette("coolwarm_r", 7),
        vmin=99,
        vmax=100
    )

    plt.ylabel('JOBS', fontsize=13)
    plt.xlabel('TIME (hours)', fontsize=13)
    plt.title('% PERCENTAGE OF COMPLETED JOBS', fontsize=16, y=-0.2)


    # Plot vertical time cutoffs
    interest_thresholds = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 20]

    for label in axes.get_xticklabels():
        if float(label.get_text()) in interest_thresholds:
            label.set_weight("bold")

    # for label in axes.get_yticklabels():
    #     label.set_size(13)

    for threshold in interest_thresholds:
        index = THRESHOLDS.index(threshold) + 1
        percentage = complete_jobs(stats_sorted, threshold)

        plt.axvline(index, color="black", linestyle='dashed')
        plt.text(index - 0.2, -0.5, percentage,
                 rotation=90, verticalalignment='bottom')

    # plt.subplots_adjust(left=3)
    plt.savefig(join(OUTPUT, 'heatmap.png'))

plot_heatmap(STATS)

# %%
plt.annotate(
    '% OF COMPLETED JOBS',
    xy=(2009, 9.5),
    xycoords='data',
    xytext=(2005, 15),
    textcoords='data',
    arrowprops=dict(arrowstyle='simple', color='#000000')
)

# %%

data = JOBS
print(data)
#%%

data_var = data[
    (data["Pipeline"] == 'CNVKIT') &
    (data["Technique"] == 'WES') &
    (data["Step Name"] == 'theta')
    ]

print(len(data_var))
# print(data_var['LogFile'].values)

# data_var.agg({
#     'Run Time': [
#         'mean',
#         'std',
#         'min',
#         'max'
#     ],
# }).to_dict()

print(data_var.agg({
    "Run Time": lambda x: np.percentile(x, 99)
}).to_dict()["Run Time"])


#%%




















eu_countries = ['BE', 'BG', 'CZ', 'DK', 'DE', 'EE', 'IE', 'GR', 'ES', 'FR', 'HR',
                'IT', 'CY', 'LV', 'LT', 'LU', 'HU', 'MT', 'NL', 'AT', 'PL', 'PT',
                'RO', 'SI', 'SK', 'FI', 'SE', 'GB']

ue = wb.download(indicator="SL.UEM.TOTL.ZS",
                 country=eu_countries, start=1991,
                 end=2014)

ue.reset_index(inplace=True)

ue.columns = ['country', 'year', 'unemployment']

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
ue_wide.sort_values(
    by=['2014', '2013', '2012', '2011', '2010'], ascending=False)
plt.figure(figsize=(10, 7))

sns.heatmap(ue_wide, cmap='YlGnBu')

plt.ylabel("")
plt.xlabel("")
plt.title("Unemployment in Europe, 1991-2013")
plt.xticks(rotation=45)

#%%
print(format_count(100))
print(format_count(10))
print(format_count(1))
print(format_count(10))
print(format_count(1100))
print(format_count(11000))
print(format_count(129455))


# %%
print(stats_index_SORTED.max())
