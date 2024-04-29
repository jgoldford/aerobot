data is stored on a publically hosted google cloud bucket:
https://storage.googleapis.com/microbe-data/aerobot/

To download, using import the "download_training_data" function in the io.py file, e.g.:

>>>from aerobot.io import download_training_data
>>>download_training_data()
