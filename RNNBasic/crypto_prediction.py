import pandas as pd
from sklearn import preprocessing
from collections import deque
import numpy as np
import random

SEQ_LEN = 60 # look at 60 min of data for prediction
FUTURE_PERIOD_PREDICT = 3 #predict the next 3 min
RATIO_TO_PREDICT = "LTC-USD"

def classify(current,future):
    if float(future) > float(current):
        return 1 #buy as future price is rising
    else:
        return  0 #sell as future price is falling

def preprocess_df(df):
    df = df.drop('future',1)
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col]= preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values: # our data should contain 
        prev_days.append(n for n in i[:-1])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days)], i[-1])

    random.shuffle(sequential_data)

main_df = pd.DataFrame()
ratios = ["BTC-USD","LTC-USD","ETH-USD","BCH-USD"]
for ratio in ratios:
    dataset = f"../Data/crypto_data/{ratio}.csv"
    df = pd.read_csv(dataset,names=["time","low","high","open","close","volume"])
    df.rename(columns={"close":f"{ratio}_close","volume":f"{ratio}_volume"}, inplace=True)

    df.set_index("time",inplace=True)
    df = df[[f"{ratio}_close",f"{ratio}_volume"]]
    print(df.head())

    #join all data into one dataset
    if len(main_df) ==0:
        main_df = df
    else:
        main_df=main_df.join(df)



main_df['future']= main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT) #organise data so that for each current price we look at we have a column that is the price 3 periods in the future

main_df['target']= list(map( classify,main_df[f"{RATIO_TO_PREDICT}_close"],main_df["future"])) # map the classify to the data so that if the future price is greater the target is 1 if the future price is less then the target is 0

print(main_df[[f"{RATIO_TO_PREDICT}_close","future","target"]].head())

times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)

