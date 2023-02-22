import get_data_function
cuts_dir = '/home/eorenst1/pipeline_hubert/data/cuts' # /path/to/

cuts_train, cuts_dev, cuts_test = get_data_function.get_data(cuts_dir)
print("train info:")
cuts_train.describe()
print("dev info:")
cuts_dev.describe()
print("test info:")
cuts_test.describe()