import get_data_function
cuts_dir = '/export/c07/efrat/pipeline_hubert/pipeline_hubert/data/cuts' # /path/to/

cuts_train_webdataset, cuts_dev, cuts_test = get_data_function.get_data(cuts_dir)
print("train info:")
cuts_train_webdataset.describe()
print("dev info:")
cuts_dev.describe()