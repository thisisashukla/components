import time

from validations import generic


def validate_dataframe(original_function):
    def wrapper_function(*args, **kwargs):

        df = original_function(*args, **kwargs)
        generic.finite_values(df)

        return df

    return wrapper_function


def time_function(original_function):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = original_function(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Function {original_function.__name__} took {round(elapsed_time/60, 2)} minutes to execute"
        )
        return result

    return wrapper
