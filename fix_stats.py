from os.path import join


def get_counts(num):
    counts = [len(line.split()) for line in open(f'stats_{num}.txt', 'r').readlines()]
    return {i: sum(1 for num in counts if num == i) for i in range(0, 11)}


def get_file_counts(file):
    counts = [len(line.split()) for line in open(file, 'r').readlines()]
    counts_dict = {i: sum(1 for num in counts if num == i) for i in range(0, 11)}
    return {k: v for k, v in counts_dict.items() if v}


# for i in range(14, 15):

#     head = '_head' if False else ''
#     line_number = 0
#     new_lines = []

#     original_file = f'fixed_stats/stats_{i}{head}.txt'
#     original_counts = get_file_counts(original_file)

#     if not original_counts.keys(): continue

#     print(f'Before -> {original_file} - {original_counts} --> sum = {sum(original_counts.values())}')
#     for line in open(original_file).readlines():
#         line_number += 1
#         # If 5 columns
#         if len(line.split()) == 5:
#             line = line.strip()
#             line += '\tnan\tnan\tnan\tnan\n'
#         # If 6 columns
#         if len(line.split()) == 6:
#             line = line.strip()
#             line += '\tnan\tnan\tnan\n'
#         # if 8 columns
#         if len(line.split()) == 8:
#             line = line.strip().split()
#             line = '\t'.join(line[:7] + ['nan'] + line[7:] + ['\n'])
#         if len(line.split()) == 9:
#             new_lines.append(line)
#     fixed_file = join("fixed_stats", f'stats_{i}{head}.txt')
#     with open(fixed_file, 'w') as new_file:
#         new_file.writelines(new_lines)

#     fixed_counts = get_file_counts(fixed_file)
#     # print(f'After -> {fixed_file} - {fixed_counts} --> sum = {sum(fixed_counts.values())}')
#     print(f'{fixed_file} -- {sum(fixed_counts.values())}/{sum(original_counts.values())}')


for i in range(17, 18):

    head = '_head' if True else ''
    line_number = 0
    new_lines = []

    original_file = join('fixed_stats', f'stats_{i}{head}.txt')
    original_counts = get_file_counts(original_file)

    print(original_file)
    if not original_counts.keys():
        continue

    print(f'Before -> {original_file} - {original_counts} --> sum = {sum(original_counts.values())}')

    fixed = 0
    for line in open(original_file).readlines():
        line_number += 1
        data = line.split()
        if float(data[5]) > 10:
            print(line)
            line = '\t'.join(data[:5] + data[6:] + [data[5]] + ['\n'])
            fixed += 1
            print(line)
        new_lines.append(line)
    fixed_file = join("fixed_stats", f'stats_{i}{head}.txt')
    with open(fixed_file, 'w') as new_file:
        new_file.writelines(new_lines)

    fixed_counts = get_file_counts(fixed_file)
    print(f'After -> {fixed_file} - {fixed_counts} --> sum = {sum(fixed_counts.values())}')
    print(f'{fixed_file} -- {sum(fixed_counts.values())}/{sum(original_counts.values())}')
    print(f'Fixed {fixed} lines')

