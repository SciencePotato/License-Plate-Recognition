import os
from util import *

# PROCESSING IMAGE
# READING IMAGE METHOD
inPath = "./data/images"
output = "/data/output/images"
confidence50, confidence80, failure = 0, 0, 0
for file in os.listdir(os.fsencode(inPath)):
    filename = os.fsdecode(file)
    sub50, temp50, temp80 = renderImageEasyOCR(inPath, filename, output)
    failure += sub50
    confidence50 += temp50
    confidence80 += temp80  
print(failure, confidence50, confidence80)