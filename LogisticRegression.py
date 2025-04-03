import kaggle 
kaggle.api.authenticate()
kaggle.api.dataset_download_files("jsphyg/weather-dataset-rattle-package",path="./csv files",unzip=True)

