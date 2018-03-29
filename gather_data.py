#%%
from os.path import join
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

from IPython.display import display, HTML

ROOT = os.getcwd()

OUTPUT = join(ROOT, 'output')

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

#%%
def get_step_name(log_file):
    file_name = log_file.split('/')[-1]
    job_name = file_name.split('.')[0]
    step_name = job_name.split('_')
    return step_name[1] if len(step_name) > 1 else None


def aggregate_jobs(head_job=True, new=False):
    frames = []
    if new:
        INPUT = 'new_collection'
        column_names = [
            "Analysis pk",
            "Technique",
            "Target Size",
            "Ref Size",
            "Pipeline",
            "LogFile",
            "Total Requested Memory",
            "Delta Memory",
            "Max Swap",
            "Run Time",
            "Step Name",
        ]
    else:
        INPUT = join(ROOT, 'fixed_stats')
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
    for pipeline_id, _ in PIPELINES:

        if head_job:
            stats_log = join(INPUT, f'stats_{pipeline_id}_head.txt')
        else:
            stats_log = join(INPUT, f'stats_{pipeline_id}.txt')

        if not os.path.isfile(stats_log): continue

        df = pd.read_csv(stats_log, sep="\t", header=None, names=column_names)

        if not df.empty:
            df['Step Name'] = df.apply(
                lambda row: get_step_name(row["LogFile"]), axis=1
            )
            frames.append(df)

    # Concatenate all pipelines dataframes
    data = pd.concat(frames)

    # Aggregate data by Time
    data_time = data[data['Run Time'].notnull()]
    data_time_grouped = data_time.groupby(["Pipeline", "Technique", "Step Name"]).agg({
        'Run Time': [
            'count',
            'mean',
            'std',
            'min',
            'max'
        ]
    }).round(1)
    data_time_grouped.columns = [
        " ".join(x) for x in data_time_grouped.columns.ravel()
    ]

    # Aggregate data by Memory
    data_memory = data[data['Total Requested Memory'].notnull() & data['Max Swap'].notnull()]
    data_memory_grouped = data_memory.groupby(["Pipeline", "Technique", "Step Name"]).agg({
        'Step Name': [
            'count'
        ],
        'Total Requested Memory': [
            'mean',
            'std',
            'min',
            'max'
        ],
        'Max Swap': [
            'mean',
            'std',
            'min',
            'max'
        ],
        'Delta Memory': [
            'mean',
            'std',
            'min',
            'max'
        ],
    }).round(1)
    data_memory_grouped.columns = [
        " ".join(x) for x in data_memory_grouped.columns.ravel()
    ]

    # Output to terminal
    # print("HEAD JOB RUNTIME" if head_job else "STEP JOBS RUNTIME")
    # print(data_time_grouped)
    # print("HEAD JOB MEMORY" if head_job else "STEP JOBS MEMORY")
    # print(data_memory_grouped)

    # Write to files
    if head_job:
        data_time_grouped.to_csv(join(OUTPUT, 'grouped_stats_time_head.txt'), sep='\t', index=False)
        data_memory_grouped.to_csv(join(OUTPUT, 'grouped_stats_memory_head.txt'), sep='\t', index=False)
    else:
        data_time_grouped.to_csv(join(OUTPUT, 'grouped_stats_time.txt'), sep='\t', index=False)
        data_memory_grouped.to_csv(join(OUTPUT,'grouped_stats_memory.txt'), sep='\t', index=False)

    with open('output/time_job.txt', 'w') as fo:
        fo.write(data_time_grouped.to_string())

    with open('output/memory_job.txt', 'w') as fo:
        fo.write(data_memory_grouped.to_string())

    return data_time, data_time_grouped, data_memory_grouped

# print('Creating Dataframes...')
# DHT = aggregate_jobs(True)
# data_total.to_csv(join(OUTPUT, 'total_head.txt'), sep='\t', index=False)
# print('Creating Dataframes...Old')
# DJTO = aggregate_jobs(False)
print('Creating Dataframes...New')
DJTN = aggregate_jobs(False)
# data_jobs.to_csv(join(OUTPUT, 'total_jobs.txt'), sep='\t', index=False)
print('Finished!')

#%% Load the Dataframes
print('Starting to read csv..')
%time total_jobs = pd.read_csv(join(OUTPUT, 'total_jobs.txt'), sep='\t')
print(f'Finished reading {len(total_jobs)} records!')


#%% Load the Dataframes
print('Starting to read csv..')
%time total_head = pd.read_csv(join(OUTPUT, 'total_head.txt'), sep='\t')
print(f'Finished reading {len(total_head)} records!')

#%%
def plot_df(data, filters, plot_var, filename, threshold=0, plot=False):

    var, ylabel, xlabel = plot_var
    data['Threshold'] = threshold
    data[var] = data[var] / (60*60)

    for column, value in filters:
        data = data[data[column] == value]
    if not len(data): return None, 0

    data_var = data[var]
    data_th = data['Threshold']

    rate_jobs_th = None

    if threshold:
        jobs_above_th = data[data[var] <= threshold]
        rate_jobs_th = round(len(jobs_above_th)/len(data_var) * 100, 1)

    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        title = f"{var} \n {' '.join([ name for _, name in filters])}"

        hist_params = {}
        if threshold:
            hist_params['range'] = [0, threshold]
        if threshold > 200:
            hist_params['logx'] = True

        ax1 = data_var.plot.hist(
            bins=10,
            title=title,
            ax=axes[0],
            grid=True,
            **hist_params,
        )
        ax1.set_xlabel(ylabel)
        ax1.set_ylabel(xlabel)

        ax2 = data_var.plot(
            style=['.'],
            title=title,
            ax=axes[1],
            color='darkorange',
            use_index=False,
            grid=True,
        )
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)

        if threshold:
            data_th.plot(
                title=f'{title} ({rate_jobs_th}% < TH)',
                ax=axes[1],
                kind="line",
                color='green',
                use_index=False,
                grid=True,
            )
        plt.tight_layout()
        plt.savefig(f'{filename}.png', bbox_inches='tight')

    return rate_jobs_th, len(data)


def plot_pipelines(data):

    thresholds = [0.1, 0.5, 1, 5, 10, 50, 100, 200, 300]
    thresholds = [0]
    stats = pd.DataFrame(
        columns=(
            ["Pipeline", "Technique", "Step Name", "Count"] +
            [str(th) for th in thresholds]
        )
    )

    # pipelines = data['Pipeline'].unique()
    # steps = data['Step Name'].unique()

    pipelines = data["Pipeline"].unique()
    for pipeline in pipelines[:]:

        techniques = data[
            data["Pipeline"] == pipeline
        ]["Technique"].unique()
        for technique in techniques:

            steps = data[
                (data["Technique"] == technique) &
                (data["Pipeline"] == pipeline)
            ]["Step Name"].unique()

            for step in steps:

                row_stats = {
                    "Pipeline": pipeline,
                    "Technique": technique,
                    "Step Name": step,
                }
                for threshold in thresholds:
                    filters = [
                        ("Pipeline", pipeline),
                        ("Technique", technique),
                        # ("Step Name", step)
                    ]
                    filename = f"{pipeline}-{technique}-{step}-{threshold}"
                    output_file = join("plots", filename)

                    plot_var = ("Run Time", "time (hours)", "# Jobs")
                    rate_jobs_th, count = plot_df(
                        data.copy(),
                        filters,
                        plot_var,
                        output_file,
                        threshold,
                        plot=True
                    )
                    row_stats[str(threshold)] = rate_jobs_th
                    row_stats["Count"] = count
                    print(f'Done {filename} = {rate_jobs_th}')
                stats = stats.append(row_stats, ignore_index=True)

    display(stats)
    stats.to_csv(join('output', f'{pipeline}.txt'), sep='\t')
    return stats

jobs = total_head.copy()
stats = plot_pipelines(jobs)

# %%
thresholds = [0.1, 1, 5, 10]

stats = pd.DataFrame(
    columns=[str(th) for th in thresholds]
)

stats = stats.append(
   {
       '0.1': 1,
       '1': 2,
       '5': 3,
       '10': 4,
   },
   ignore_index=True
)
stats = stats.append(
    {
        '0.1': 1,
        '1': 2,
        '5': 3,
        '10': 4,
    },
    ignore_index=True
)
print(stats)

# %%
from multiprocessing import Pool

def load_csv():
    reader = pd.read_csv(
        join(OUTPUT, 'total_head.txt'),
        sep='\t',
        chunksize=1000
    )
    pool = Pool(4)


    funclist = []
    for df in reader:
        funclist.append(df)

    total = pd.DataFrame()
    print(len(funclist))
    for f in funclist:
        total = total.append(f, ignore_index=True)
        # result += f.get()  # timeout in 10 seconds
    # print(f"There are {result} rows of data")
    return total

test_csv = 0
print('Starting to read csv..')
%time total_head = load_csv()
print(f'Finished reading {len(total_head)} records!')


# %%

with open('output/stats_percentage_jobs.txt', 'w') as fo:
    fo.write(stats.to_string())

# %%

with open('output/summary_memory_job.txt', 'w') as fo:
    fo.write(DJT[2].to_string())

with open('output/summary_time_job.txt', 'w') as fo:
    fo.write(DJT[1].to_string())

with open('output/summary_memory_head.txt', 'w') as fo:
    fo.write(DHT[2].to_string())

with open('output/summary_time_head.txt', 'w') as fo:
    fo.write(DHT[1].to_string())


# %%


DJT[2].to_csv('output/csv_summary_memory_job.csv')
DJT[1].to_csv('output/csv_summary_time_job.csv')
# DHT[2].to_csv('output/csv_summary_memory_head.csv')
# DHT[1].to_csv('output/csv_summary_time_head.csv')

# %%
data = DHT[0]
data = data[(data['Pipeline'] == 'QC_DATA') & (data['Technique'] == 'WTA')]
data = data.drop('BAM Size kb', axis=1)
data = data.drop('Technique', axis=1)
data = data.drop('LogFile', axis=1)
data = data.drop('Step Name', axis=1)

print(data.columns)

DFJ = pd.merge(DJTO[0], DJTN[0])

DFJ = DFJ.drop("BAM Size kb", axis=1)
print(len(DFJ))
print(DFJ.columns)

# %%

def aggregate_data(data):
    # Aggregate dat by (time and memory) per Gb
    data['Time per 100Gb'] = data.apply(
        lambda row: 100/3600 * row["Run Time"] / row["Target Size"], axis=1
    )
    data['Memory per 100Gb'] = data.apply(
        lambda row: 100 * row["Max Swap"] / row["Target Size"], axis=1
    )

    data_filtered = data[
        data['Time per 100Gb'].notnull() & data['Memory per 100Gb'].notnull()
    ]
    data_grouped = data_filtered.groupby(["Pipeline", "Technique", "Step Name"]).agg({
        'Step Name': [
            'count'
        ],
        'Time per 100Gb': [
            'mean',
            'std',
            'min',
            'max'
        ],
        'Memory per 100Gb': [
            'mean',
            'std',
            'min',
            'max'
        ],
    }).round(1)
    data_grouped.columns = [
        " ".join(x) for x in data_grouped.columns.ravel()
    ]
    return data, data_grouped

df_data = aggregate_data(DFJ)



# print(df_data[2])
df_data[1].to_csv('output/dataframe_grouped.csv')
print('Finished!')

# %%
with open('output/dataframe_sizes.txt', 'w') as fo:
    fo.write(DFJ.to_string())

# %%
print("Hello)
