import os
import gdown
from os.path import exists

current_directory = os.path.split(os.getcwd())[1]

url = "https://drive.google.com/file/d/1rwqa4kK-iDfIc8jji6euEfs2rt7_qx_s/view?usp=sharing"

output_file = "flash-free.zip"

if current_directory != "holmes":
    print("This is the wrong directory. Please run this script in the folder data/flash-holmes")
else:
    gdown.download(url=url, output=output_file, fuzzy=True)

    if not exists("holmes-free.zip"):
        print("Download failed. Please try again.")
    else:
        os.system("unzip holmes-free.zip")