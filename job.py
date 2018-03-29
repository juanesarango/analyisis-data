"""
Leukbot queries
"""

from itertools import islice
from os import listdir
from os.path import join, isdir


import pandas as pd
import numpy as np

import leuktools as lk

analysis_count = [
    (1, 'BATTENBERG', 274),
    (2, 'BRASS', 216),
    (3, 'CAVEMAN', 10625),
    (4, 'CNVKIT', 7017),
    (5, 'FLT3ITD', 6757),
    (6, 'PINDEL', 10705),
    (7, 'RNACALLER', 738),
    (8, 'MERGE_TABLES', 24),
    (9, 'ANNOT_CAVEMAN', 14081),
    (10, 'ANNOT_PINDEL', 10848),
    (11, 'ANNOT_CAVEMAN', 14081),
    (12, 'ANNOT_PINDEL', 10848),
    (13, 'PTD', 3),
    (14, 'QC_DATA', 11483),
    (15, 'FACETS', 487),
    (16, 'RNAFUSIONS', 29),
    (17, 'CONPAIR', 29)
    ]



# Parse utils
"""
    CPU time :                                   2725.44 sec.
    Total Requested Memory :                     8.00 GB
    Delta Memory :                               -
    Max Processes :                              11
    Max Threads :                                12
    Run time :                                   2732 sec.
    Turnaround time :                            2731 sec.
"""

def parse_value(value):
    try:
        return float(value.strip())
    except Exception:
        return np.nan

# Get key: Value from logs
def parse_line(line, before_text, after_text):
    _, value = line.replace(after_text, '').split(before_text)
    return parse_value(value)

def parse_time(line):
    _, time = line.replace(' ', '').replace('sec.', '').split('Runtime:')
    return parse_value(value)

def parse_max_memory(line):
    _, memory = line.replace(' ', '').replace('GB', '').split('MaxSwap:')
    return parse_value(value)

def parse_delta_memory(line):
    _, memory = line.replace(' ', '').replace('GB', '').split('DeltaMemory:')
    return parse_value(value)

# Parse Jobs Functions

def parse_job_files(analysis, log_files, df_jobs):
    df_jobs = collect_jobs_stats(analysis, log_files, df_jobs)
    return df_jobs

fields_to_parse = [
    ("RunTime:", 'sec.'),
    ("DeltaMemory:", 'GB'),
    ("MaxSwap:", 'sec.'),
    ("TotalRequestedMemory", 'sec.'),
    ]

def collect_jobs_stats(analysis, log_files, df_jobs):

    for log_file in log_files:
        method = analysis.as_target_objects[0].technique.method
        job_stats = {
                'Analysis pk': analysis.pk,
                'Technique': method,
                'BAM Size kb': analysis.get_outdir_usage(),
                'Pipeline': analysis.name,
                'LogFile': log_file,
                }
        with open(log_file, 'r') as file:
            job_time = None
            job_memory = None
            try:
                for line in file:
                    line = line.replace(' ', '')
                    field = [(field_name, units) for field_name, units in fields_to_parse if field_name in line]
                    if len(field) == 1:
                        field_name, units = field[0]
                        job_stats[field_name] = parse_line(line, field_name, units)
            except Exception:
                pass
            df_jobs = df_jobs.append(job_stats, ignore_index=True)
    return df_jobs

df_jobs = pd.DataFrame()

for analysis in all_analysis:

    logs_dir = join(analysis.outdir, "logs_lsf")

    if isdir(logs_dir):
        log_files = [
            join(logs_dir, log_file)
            for log_file in listdir(logs_dir)
            if '.logs' in log_file
            ]

        df_jobs = parse_job_files(
            analysis,
            log_files,
            df_jobs,
            )
