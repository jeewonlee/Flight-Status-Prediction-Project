import pandas as pd
import numpy as np

#combine 12 month data
def concat_df():
    df = pd.read_csv('data/201601.csv')
    for num in range(2, 13):
        if num < 10:
            temp = pd.read_csv("data/20160"+str(num)+".csv")
        else:
            temp = pd.read_csv("data/2016"+str(num)+".csv")
        df = pd.concat([df, temp])
    df.fillna(0, inplace=True)
    new_df = df[['MONTH', 'DAY_OF_WEEK', 'CARRIER','ORIGIN', 'DEST', 'DEP_TIME', 'ARR_DELAY_NEW','DISTANCE_GROUP']]
    return new_df

#create departure time groups
def group_dep_time(df):
    dep_label = ['00-04','04-08','08-12','12-16','16-20','20-24']
    df['DEP_TIME_BINS'] = pd.cut(df.DEP_TIME,[0.0, 400.0, 800.0, 1200.0, 1600.0, 2000.00, 2400.00] ,labels=dep_label)
    df.pop('DEP_TIME')
    return df

def create_label_dummies(df):
    #Create label: 1 = delayed, 0 = on-time
    df['label'] = df['ARR_DELAY_NEW'].map(lambda x: 1 if x > 0 else 0)
    print "After creating label # of columns: ", len(df.columns)

    #create Month dummies
    df_month = pd.get_dummies(df.MONTH)
    df_month.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    df = pd.concat([df, df_month], axis=1)
    df.pop('MONTH')
    print "\nMonth # of columns: ", len(df_month.columns)
    print "Currnet df # of columns: ", len(df.columns)

    #create DOW dummies
    df_dow = pd.get_dummies(df.DAY_OF_WEEK)
    df_dow.columns = ['Mon','Tue','Wed','Thr','Fri','Sat','Sun']
    df = pd.concat([df, df_dow], axis=1)
    df.pop('DAY_OF_WEEK')
    print "\nDOW # of columns: ", len(df_dow.columns)
    print "Currnet df # of columns: ", len(df.columns)

    #create CARRIER dummies
    df_carrier = pd.get_dummies(df.CARRIER)
    df = pd.concat([df, df_carrier], axis=1)
    df.pop('CARRIER')
    print "\nCARRIER # of columns: ", len(df_carrier.columns)
    print "Currnet df # of columns: ", len(df.columns)

    #create Origin dumimies
    df.ORIGIN = df.ORIGIN.map(lambda x: "orig_"+x)
    df_origin = pd.get_dummies(df.ORIGIN)
    df = pd.concat([df, df_origin], axis=1)
    df.pop('ORIGIN')
    print "\nOrigin # of columns: ", len(df_origin.columns)
    print "Currnet df # of columns: ", len(df.columns)

    #create Time group dumimies
    df_dep_time = pd.get_dummies(df.DEP_TIME_BINS)
    df = pd.concat([df, df_dep_time], axis=1)
    df.pop('DEP_TIME_BINS')
    print "\nDeparture Time # of columns: ", len(df_dep_time.columns)
    print "Currnet df # of columns: ", len(df.columns)

    #create Dest dumimies
    # df.DEST = df.DEST.map(lambda x: "dest_"+x)
    # df_dest = pd.get_dummies(df.DEST)
    # df = pd.concat([df, df_dest], axis=1)
    df.pop('DEST')
    # print "\nDest # of columns: ", len(df_dest.columns)
    print "\nPop Dest, Currnet df # of columns: ", len(df.columns)

    #create Distance group dummies
    df_distance_group = pd.get_dummies(df.DISTANCE_GROUP)
    df_distance_group.columns = ["Less Than 250 Miles","250-499 Miles","500-749 Miles","750-999 Miles","1000-1249 Miles","1250-1499 Miles","1500-1749 Miles"
,"1750-1999 Miles","2000-2249 Miles","2250-2499 Miles","2500 Miles and Greater"]
    df = pd.concat([df, df_distance_group], axis=1)
    df.pop('DISTANCE_GROUP')
    print "\nDistance group # of columns: ", len(df_distance_group.columns)
    print "Currnet df # of columns: ", len(df.columns)
    return df

#create dataset for modeling
def create_dataset():
    df = concat_df()
    print "After concatenating # of columns: ", len(df.columns)
    df = group_dep_time(df)
    print "After creating departure group # of columns: ", len(df.columns)
    df = create_label_dummies(df)
    return df
