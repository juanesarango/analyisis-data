#!/usr/bin/env python

from itertools import islice
from glob import glob
from os import listdir
from os.path import join, isdir
import json
import os
import subprocess
import sys

import pandas as pd
import numpy as np

import leuktools as lk

"""
Info of current analysis in Leukgen

    analysis_count = [
        (1, 'BATTENBERG', 274),
        (2, 'BRASS', 216),
        (3, 'CAVEMAN', 10789),
        (4, 'CNVKIT', 7333),
        (5, 'FLT3ITD', 7073),
        (6, 'PINDEL', 10869),
        (7, 'RNACALLER', 738),
        (8, 'MERGE_TABLES', 24),
        (9, 'ANNOT_CAVEMAN', 3462),
        (10, 'ANNOT_PINDEL', 143),
        (11, 'ANNOT_CAVEMAN', 10783),
        (12, 'ANNOT_PINDEL', 10869),
        (13, 'PTD', 3),
        (14, 'QC_DATA', 11483),
        (15, 'FACETS', 487),
        (16, 'RNAFUSIONS', 29),
        (17, 'CONPAIR', 29)
    ]

Fields found in a LSF logs file for a given analysis:

    CPU time :                                   2725.44 sec.
    Total Requested Memory :                     8.00 GB
    Delta Memory :                               -
    Max Processes :                              11
    Max Threads :                                12
    Run time :                                   2732 sec.
    Turnaround time :                            2731 sec.

CSV headers:
    Analysis pk
    Technique
    BAM Size kb
    Pipeline
    LogFile
    Run Time
    Total Requested Memory
    Delta Memory
    Max Swap
"""


# Utils to parse values
def parse_value(value):
    try:
        return float(value.strip())
    except Exception:
        return np.nan


def get_input_file_sizes(analysis):
    targets_size = sum([
        os.stat(target.bampath).st_size
        for target in analysis.as_target_objects
    ]) / 1024**3
    refs_size = sum([
        os.stat(ref.bampath).st_size
        for ref in analysis.as_reference_objects
    ]) / 1024**3
    return targets_size, refs_size


def parse_line(line, before_text, after_text):
    _, value = line.replace(after_text, '').split(before_text)
    return parse_value(value)


# Parse Jobs Functions
def parse_dict(python_dict):
    return '\t'.join([str(value) for value in python_dict.values()]) + '\n'


def get_and_store_stats(analysis, log_files, output_file):

    fields_to_parse = [
        ("Runtime:", 'sec.'),
        ("TotalRequestedMemory:", 'GB'),
        ("DeltaMemory:", 'GB'),
        ("MaxSwap:", 'GB'),
    ]

    for log_file in log_files:
        method = analysis.as_target_objects[0].technique.method
        tumor_size, normal_size = get_input_file_sizes(analysis)
        job_stats = {
            'Analysis pk': analysis.pk,
            'Technique': method,
            'Target BAM GB': tumor_size,
            'Ref BAM GB': normal_size,
            'Pipeline': analysis.name,
            'LogFile': log_file,
        }
        with open(log_file, 'r') as file:
            try:
                for line in file:
                    line = line.replace(' ', '')
                    field = [(field_name, units) for field_name,
                             units in fields_to_parse if field_name in line]
                    if len(field) == 1:
                        field_name, units = field[0]
                        job_stats[field_name] = parse_line(
                            line, field_name, units)
            except Exception:
                pass
            output_file.write(parse_dict(job_stats))


def get_last_analysis(output_file):
    try:
        last_line = subprocess.check_output(['tail', '-1', output_file])
        return int(last_line.decode().strip().split('\t')[0])
    except Exception:
        return 0


# Main Function
def get_analysis_info(pipeline_id):

    output_file_path = join('new_collection', f'stats_{pipeline_id}_head.txt')
    initial_pk = get_last_analysis(output_file_path)

    filters = [
        ('pk__gt', initial_pk),
        ('pipeline__pk', pipeline_id)
    ]
    all_analysis = lk.get_analyses(filters=filters)

    with open(output_file_path, "a") as output_file:
        for analysis in all_analysis:
            head_job_logs = glob(join(analysis.outdir, "head_job*.log"))
            if head_job_logs:
                print(f'Logs: {head_job_logs}.')
                try:
                    get_and_store_stats(
                        analysis,
                        head_job_logs,
                        output_file
                    )
                except Exception as error:
                    print(
                        f'Analysis {analysis.pk}-{analysis.name} failed. '
                        f'Logs: {head_job_logs}. '
                        f'Error: {error}'
                    )


if __name__ == "__main__":

    if len(sys.argv) == 2:
        pipeline_id = int(sys.argv[1])
        get_analysis_info(pipeline_id)
    else:
        print('Pass the ID of the Analyisis Pipeline')
