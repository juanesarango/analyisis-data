import pandas as pd


def mem_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:  # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


def memory_optimization(gl):
    # Memory Info
    for dtype in ['float', 'int', 'object']:
        selected_dtype = gl.select_dtypes(include=[dtype])
        mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
        mean_usage_mb = mean_usage_b / 1024 ** 2
        print("Average memory usage for {} columns: {:03.2f} MB".format(
            dtype, mean_usage_mb))

    # Optimize Integers
    gl_int = gl.select_dtypes(include=['int'])
    converted_int = gl_int.apply(pd.to_numeric, downcast='unsigned')

    print(mem_usage(gl_int))
    print(mem_usage(converted_int))

    compare_ints = pd.concat([gl_int.dtypes, converted_int.dtypes], axis=1)
    compare_ints.columns = ['before', 'after']
    compare_ints.apply(pd.Series.value_counts)
    print(compare_ints)

    # Optimize Float
    gl_float = gl.select_dtypes(include=['float'])
    converted_float = gl_float.apply(pd.to_numeric, downcast='float')

    print(mem_usage(gl_float))
    print(mem_usage(converted_float))

    compare_floats = pd.concat(
        [gl_float.dtypes, converted_float.dtypes], axis=1)
    compare_floats.columns = ['before', 'after']
    compare_floats.apply(pd.Series.value_counts)

    print(compare_floats)

    # Asign Optimized Columns
    optimized_gl = gl.copy()

    optimized_gl[converted_int.columns] = converted_int
    optimized_gl[converted_float.columns] = converted_float

    print(mem_usage(gl))
    print(mem_usage(optimized_gl))

    return optimized_gl

# % time total_jobs2 = total_jobs.copy()
# % time total_jobs_opt = memory_optimization(total_jobs2)
