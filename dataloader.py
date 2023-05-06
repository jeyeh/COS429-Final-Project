def Dataloader():
    import os
    import zipfile
    import pandas as pd
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

    # using code from https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/
    # and https://stackoverflow.com/questions/3451111/unzipping-files-in-python

    # UNZIP ALL ZIP FILES
    directory = 'word_data'
    
    # iterate over files in
    # that directory

    zipfiles = []
    directories = []
    dir_contents = os.listdir(directory)

    for filename in dir_contents:
            if('.zip' in filename):
                zipfiles.append(filename)
                with zipfile.ZipFile('./word_data/' + filename, 'r') as zip_ref:
                    print("Loading " , filename)
                    zip_ref.extractall(directory + '/' + filename[0:-4])
                    directories.append(directory + '/' + filename[0:-4])

    # with code from https://www.geeksforgeeks.org/convert-excel-to-csv-in-python/

    # read an excel file and convert 
    # into a dataframe object

    imgs = []
    labels = []


    for directory in directories:
        df = pd.DataFrame(pd.read_excel(directory + ".xlsx"))


        df = df.to_numpy()
        df = df[:,1:]
    #     print(df)
        
        for ind, entry in enumerate(df):
            imgPath = directory + '/' + directory[-3:] + '/' + entry[0]
            # print(imgPath)
            imgs.append(cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE))
            labels.append(entry[1])

    return(imgs, labels)
            
            