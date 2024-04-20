import pandas as pd

df1 = pd.read_csv("./data/test_data.csv")
df2 = pd.read_csv("./data/train_data.csv")


for row in df1.iterrows():
    print(row[1]["filename"].split(".")[0])

    norm_xmin = row[1]["xmin"]/row[1]["width"]
    norm_ymin = row[1]["ymin"]/row[1]["height"]
    norm_xmax = row[1]["xmax"]/row[1]["width"]
    norm_ymax = row[1]["ymax"]/row[1]["height"]

    with open("./data/labels/" + str(row[1]["filename"].split(".")[0])+ ".txt", "w") as f:
        f.write("0 " + str(norm_xmin) + " " + str(norm_ymin) + " " + str(norm_xmax) + " " + str(norm_ymax) + "\n")


for row in df2.iterrows():
    print(row[1]["filename"].split(".")[0])

    norm_xmin = row[1]["xmin"]/row[1]["width"]
    norm_ymin = row[1]["ymin"]/row[1]["height"]
    norm_xmax = row[1]["xmax"]/row[1]["width"]
    norm_ymax = row[1]["ymax"]/row[1]["height"]

    with open("./data/labels/" + str(row[1]["filename"].split(".")[0])+ ".txt", "w") as f:
        f.write("0 " + str(norm_xmin) + " " + str(norm_ymin) + " " + str(norm_xmax) + " " + str(norm_ymax) + "\n")
