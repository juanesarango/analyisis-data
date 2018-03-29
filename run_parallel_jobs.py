import subprocess


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

if __name__ == "__main__":
    for pipeline_id, _ in PIPELINES:
        cmd = [
            "bsub",
            "-We", "60",
            "-n", "8",
            "-M", "10",
            "-R", "rusage[mem=10]",
            '-o', f'logs/new-collect-stats-pipeline-{pipeline_id}.%J.log',
            f'python job_pipeline_collection.py {pipeline_id}'
        ]
        print(' '.join(cmd))
        output = subprocess.check_output(cmd)
        print(output)


