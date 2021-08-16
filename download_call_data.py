#! /usr/bin/env python3


import pandas as pd
#Load Call Volume Data
from sodapy import Socrata

client = Socrata("data.seattle.gov", None)
print("Reaching Client and Fetching Data ...")
results = client.get("kzjm-xkqj", where="datetime BETWEEN '2004-01-01T00:00:00.000' AND '2021-08-15T23:59:59.000'", limit=1600000)
print("Download Complete, Writing out Dataframe into CSV file")

# Convert to pandas DataFrame
main_df = pd.DataFrame.from_records(results)
main_df.to_csv("seattle_sos.csv", index=False)

print("Done! Now you can run the main executable ./main.py")