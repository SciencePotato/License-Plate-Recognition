import os
from util import *

# PROCESSING IMAGE
# READING IMAGE METHOD
inPath = "./data/images"
output = "/data/output/images"
for file in os.listdir(os.fsencode(inPath)):
    filename = os.fsdecode(file)
    renderImage(inPath, filename, output)