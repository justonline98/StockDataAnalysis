#TODO Improvementsw
#1. fix DB update to target each stock instead of all
#2. parallel methods
#3. include other stock index with more values
    #sp1500
    #dowjones
    #https://en.wikipedia.org/wiki/List_of_stock_market_indices


import pandas as pd
import pandas_ta as pta
import yfinance as yf
from datetime import datetime, timedelta
import requests
import json
from dateutil.relativedelta import relativedelta
import os, shutil
import threading
import sqlite3
import queue
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor, XGBClassifier
import tensorflow as tf
import seaborn as sns
import io
import base64



#sequential model
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 8)

def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]
####################Feature Engineering Functions#################
def MAGreater(Lower,Higher):
    if(Lower > Higher):
        return 1
    else:
        return 0

def BuyPercent(first,second,percent) -> bool:
    target = first + (first*percent)
    if(target > second):
        return 0
    else:
        return 1
    
def RSIgood(val,target):
    if(val > target):
        return 0
    else:
        return 1
    
def fig_to_base64(fig:plt.Figure):
    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)

    return base64.b64encode(img.getvalue())
#####################################################################

##########################eval tools################################# 
  
def SellPattern(data:pd.DataFrame, ammount:float) -> pd.DataFrame:
    modifieddata = data.copy()
    modifieddata["Sell"] = False

    SellPrice = 0
    for index,row in modifieddata.iterrows():
        PriceCheck = row["Close"]*(1-ammount)
        if PriceCheck > SellPrice:
            SellPrice = PriceCheck
        if row["Close"] <  SellPrice:
            modifieddata.at[index,"Sell"] = True
            SellPrice = row["Close"]*(1.0-ammount)
    return modifieddata


def OwnPattern(data:pd.DataFrame) -> pd.DataFrame:
    modifieddata = data.copy()
    modifieddata["Own"] = False
    pastValue = False
    for index,row in modifieddata.iterrows():
        curvalue = pastValue
        if row["Buy"] == True:
            curvalue = True
        if row["Sell"] == True:
            curvalue = False
        modifieddata.at[index,"Own"] = curvalue
        pastValue=curvalue
        
    return modifieddata

def TotalValue(data:pd.DataFrame, ammount:float) -> pd.DataFrame:
    modifieddata = data.copy()
    modifieddata["Value"] = ammount
    CurrentValue=ammount
    Shares=0
    for index,row in modifieddata.iterrows():
        if row["Own"] == True and Shares == 0:
            #own none buy
            Shares = CurrentValue/row["Close"]
        elif row["Own"] == False and Shares != 0:
            #sell signal our owned shares
            CurrentValue = Shares*row["Open"]
            Shares = 0

        #just calculate value for day
        if Shares != 0:
            CurrentValue = Shares*row["Close"]

        modifieddata.at[index,"Value"] = CurrentValue
        
    return modifieddata
#####################################################################

class StockStats: 
    def __init__(self,Ticker) -> None:
        #information for NYSE call to get companies list
        self.Ticker = Ticker
        self.Gain = 0
        self.price = 0
        #print('Model Performance')
        #print('Train RMSE = {:0.2f}.'.format(self.train_rmse))
        #print('Train MAE = {:0.2f}.'.format(self.train_mae))
        #print('Test RMSE = {:0.2f}.'.format(self.test_rmse))
        #print('Test MAE = {:0.2f}.'.format(test_mae))
        self.train_rmse=0
        self.train_mae=0
        self.test_rmse =0
        self.test_mae=0

        #plots
        self.Averages = ''
        self.confusion_matrix = ''
        self.TestBuySignals = ''
        self.BuySellGraph = ''
        self.TestOwnedGraph = ''
        self.totalOwnedGraph = ''

        #html elements
        self.Padding = "display: block;margin-left: auto;margin-right: auto;"
        self.ImageWidth = "width:75%;"
        self.LineBreak = "<hr>"


    def GenerateHTMLPage(self) -> str:
        my_html = "<hr>"
        my_html = my_html + "<h1>"+ self.Ticker +"</h1>"
        my_html = my_html + "<p>Percent Return: "+ str((self.Gain-100)/100)  + "/100</p>"
        my_html = my_html + "<p>Ending Sim Return: "+ str(self.Gain) +"</p>"
        my_html = my_html + "<p>Current Price: "+ str(self.price) +"</p>"
        my_html = my_html + '<img style=\'{}\'src="data:image/png;base64, {}">'.format(self.Padding,self.confusion_matrix.decode('utf-8'))
        my_html = my_html + '<img style=\'{}\'src="data:image/png;base64, {}">'.format(self.ImageWidth+self.Padding,self.Averages.decode('utf-8'))
        my_html = my_html + '<img style=\'{}\'src="data:image/png;base64, {}">'.format(self.ImageWidth+self.Padding,self.TestBuySignals.decode('utf-8'))
        my_html = my_html + '<img style=\'{}\'src="data:image/png;base64, {}">'.format(self.ImageWidth+self.Padding,self.BuySellGraph.decode('utf-8'))
        my_html = my_html + '<img style=\'{}\'src="data:image/png;base64, {}">'.format(self.ImageWidth+self.Padding,self.TestOwnedGraph.decode('utf-8'))
        #my_html = my_html + '<img style=\'{}\'src="data:image/png;base64, {}">'.format(self.ImageWidth+self.Padding,self.totalOwnedGraph.decode('utf-8'))
        my_html = my_html + "<hr>"
        return my_html

    def evaluateRegression(self, model, train_features, train_labels, val_features, val_labels):
        train_predictions = model.predict(train_features)
        val_predictions = model.predict(val_features)
        self.train_rmse = math.sqrt(mean_squared_error(train_labels, train_predictions))
        self.train_mae=mean_absolute_error(train_labels, train_predictions)
        self.test_rmse = math.sqrt(mean_squared_error(val_labels, val_predictions))
        self.test_mae=mean_absolute_error(val_labels, val_predictions)
        cm = confusion_matrix(val_labels, val_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        ConfusionPlot = disp.plot().figure_
        self.confusion_matrix = fig_to_base64(ConfusionPlot)
        return train_predictions, val_predictions

    def MovingAverages(self,dfData: pd.DataFrame):
        plt.figure(figsize=(20,7))
        plt.plot(dfData.index,dfData['Close'],color='black',label='Close')
        plt.plot(dfData.index,dfData['ma10'],color='red',label='MA10')
        plt.plot(dfData.index,dfData['ma20'],color='green',label='MA20')
        plt.plot(dfData.index,dfData['ma30'],color='blue',label='MA30')
        plt.plot(dfData.index,dfData['ma40'],color='purple',label='MA40')
        plt.plot(dfData.index,dfData['ma50'],color='pink',label='MA50')
        plt.plot(dfData.index,dfData['ma60'],color='orange',label='MA60')
        plt.plot(dfData.index,dfData['ma70'],color='goldenrod',label='MA70')
        plt.legend()
        self.Averages = fig_to_base64(plt)
        plt.clf()
        plt.close()

    def ComparePredictions(self,Predvtest: pd.DataFrame):
        plt.figure(figsize=(20,7))
        plt.plot(Predvtest.index,Predvtest['CloseBuy'],color='blue',label='actual')
        plt.plot(Predvtest.index,Predvtest["pred"],color='red',label='predicted')
        plt.legend()
        self.TestBuySignals = fig_to_base64(plt)
        plt.clf()
        plt.close()

    def SellBuyGraphCreation(self, DfData:pd.DataFrame):
        plt.figure(figsize=(20,7))
        plt.plot(DfData.index,DfData['Close'],color='blue',label='actual')
        for i, row in DfData.iterrows():
            if row["Sell"] != False:
                plt.axvline(x = i, color="red")
            if row["Buy"] != False:
                plt.axvline(x = i, color="green")
        plt.legend()
        self.BuySellGraph = fig_to_base64(plt)
        plt.clf()
        plt.close()
        
    def OwnedGraphCreation(self, DfData:pd.DataFrame):
        plt.figure(figsize=(20,7))
        plt.plot(DfData.index,DfData['Close'],color='blue',label='actual')
        for i, row in DfData.iterrows():
            if row["Own"] != False:
                plt.axvline(x = i, color="green")
        plt.legend()
        self.TestOwnedGraph = fig_to_base64(plt)
        plt.clf()
        plt.close()

    def ValueGraphCreation(self, DfData:pd.DataFrame):
        plt.plot(DfData.index,DfData['Value'],color='blue',label='value')
        plt.legend()
        self.totalOwnedGraph = fig_to_base64(plt)
        plt.clf()
        plt.close()


class StockAutomation:

    def __init__(self) -> None:
        #information for NYSE call to get companies list
        self.NYSE_URL = "https://www.nyse.com/api/quotes/filter"
        self.SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        self.SP600_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
        self.SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        self.NYSE_payload = json.dumps({
        "instrumentType": "EQUITY",
        "pageNumber": 1,
        "sortColumn": "NORMALIZED_TICKER",
        "sortOrder": "ASC",
        "maxResultsPerPage": 7502,
        "filterToken": ""
        })
        self.NYSE_headers = {
        'Referer': 'https://www.nyse.com/listings_directory/stock',
        'Origin': 'https://www.nyse.com',
        'Content-Type': 'application/json',
        'Cookie': 'BIGipServernyse.com=1174809866.16651.0000'
        }
        self.NYSE_list = []
        self.SP500_list = []
        self.DataPath = 'CSVs\Stock_data\RawData.csv'
        self.DataTable = "StockTable"
        self.Threadcount = 4
        self.Threads = []
        self.TickerList = []
        self.years_back = 2
        self.ColHeaders = [
            "PK",
            "Ticker",
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume"
        ]
        
        ####prediction Values
        self.TargetPercentGain = 0.07
        self.DaysAhead = -3
        self.TestSize = 0.2
        self.ActualPredSize = 0.08 #20 Days for pred/ 252 trading days
        self.train_features = [
            # Initial Data
            #"Open",
            #"High",
            #"Low",
            #"Volume",
            #"Close",

            # Moving averages
            #"ma5",
            #"ma8",
            #"ma13",
            #"ma10",
            #"ma20",
            #"ma30",
            #"ma40",
            #"ma50",
            #"ma60",
            #"ma70",

            #Bool Moving Averages
            "5v10",
            "5v20",
            "5v30",
            "5v40",
            "5v50",
            "5v60",
            "5v70",

            "10v20",
            "10v30",
            "10v40",
            "10v50",
            "10v60",
            "10v70",

            "20v30",
            "20v40",
            "20v50",
            "20v60",
            "20v70",

            "30v40",
            "30v50",
            "30v60",
            "30v70",
            
            #RSI
            "rsi",
            "rsiBool"
            
            ]

        self.predFeature = 'CloseBuy'
        self.percentsell = 0.1
        self.StartTestMoney = 1000
        self.StockStats = []
        self.BuySignalsinPast = 3 #buy signal in past seven trading days

    #gets only the common stocks from NYSE since that is what is in robin hood
    def GetNYSECompanies(self):
        response = requests.request("POST", self.NYSE_URL, headers=self.NYSE_headers, data=self.NYSE_payload)
        data = pd.DataFrame(json.loads(response.text),dtype=str)
        data = data.loc[data["instrumentType"] == "COMMON_STOCK"]
        data.to_csv("CSVs/NYSE.csv", index=False)
        self.NYSE_list = data["symbolTicker"].to_list()
        return self.NYSE_list

    #gets sp500 stocks
    def GetSP500Companies(self):
        self.SP500_list = pd.read_html(self.SP500_URL)[0]["Symbol"].to_list()
        return self.SP500_list, "sp500"

    #gets sp500 stocks
    def GetSP600Companies(self):
        self.SP500_list = pd.read_html(self.SP600_URL)[0]["Symbol"].to_list()
        return self.SP500_list, "sp600"
    
        #gets sp500 stocks
    def GetSP400Companies(self):
        self.SP500_list = pd.read_html(self.SP400_URL)[0]["Symbol"].to_list()
        return self.SP500_list, "sp400"

    #delete all stock data
    def DeleteOldData(self):
        for filename in os.listdir(self.Data_Folder):
            file_path = os.path.join(self.Data_Folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


    #returns all data for given list of stocks
    def GetStockData(self, ListOfTickers:[str], TableName:str):
        dfOrig = pd.DataFrame()
        #if(os.path.isfile(self.DataPath) == False):
        if(self.CheckTableExists(TableName) == False):
            print("Data File Doesnt Exist: getting data from " + str(self.years_back) + " Years ago")
            Start_Date = (datetime.now() - relativedelta(years=self.years_back)).strftime("%Y-%m-%d")
            print("Getting Data starting at Date: " + Start_Date)
            dfOrig = yf.download(ListOfTickers, group_by='ticker', start=Start_Date, interval="1d")
            dfOrig = self.FormatData(dfOrig)
        else:
            print("File Exists: running update code")
            #reading file and getting date for last update
            #dfOrig = pd.read_csv(self.DataPath)
            dfOrig = self.GetAllData(TableName)
            dfOrig["Date"] = pd.to_datetime(dfOrig["Date"],format="%Y-%m-%d")
            UpdateDate = dfOrig["Date"].to_list()
            UpdateDate.sort(reverse=True)
            UpdateDate = (UpdateDate[0]).strftime("%Y-%m-%d")

            if(UpdateDate != datetime.now().strftime("%Y-%m-%d")):
                #gettin data and formatting
                print("Updating starting at this date: " + UpdateDate)
                dfNewData = yf.download(ListOfTickers, group_by='ticker', start=UpdateDate, interval="1d")
                dfNewData = self.FormatData(dfNewData)
                dfNewData = dfNewData.set_index("PK")
                dfOrig = dfOrig.set_index("PK")

                #updating and resetting index
                dfOrig = pd.concat([dfOrig, dfNewData[~dfNewData.index.isin(dfOrig.index)]])
                dfOrig.update(dfNewData)
                dfOrig.sort_index(inplace=True)
                dfOrig = dfOrig.reset_index()

                #removing those older than date
                dfOrig = dfOrig[pd.to_datetime(dfOrig['Date']) >= (datetime.now()- relativedelta(years=self.years_back))]
            else:
                print("No update, Data is current up to " + UpdateDate)

        #dfOrig.to_csv(self.DataPath,index=False)
        self.CreateReplaceTable(dfOrig,TableName)

        
    #method turns multiindexing into single index dataframe
    #pk set as ticker+date
    #ticker column added for sorting later on
    def  FormatData(self, StockData: pd.DataFrame) -> pd.DataFrame:
        FinalTable = pd.DataFrame(columns=self.ColHeaders)
        for ticker in pd.Series(StockData.columns.get_level_values(0)).drop_duplicates().tolist():
            SingleStockData = StockData[str(ticker)].reset_index(level=[0])
            SingleStockData["PK"] = StockData.index.astype(str) + ticker
            SingleStockData["Ticker"] = ticker
            FinalTable = pd.concat([FinalTable,SingleStockData])
            print(ticker)
        return FinalTable
    
    #check if table exists
    def CheckTableExists(self, name) ->bool:
        con = sqlite3.connect('CSVs\Stock_data\RawData.db')
        cur = con.cursor()
        listOfTables = cur.execute(f"""SELECT name FROM sqlite_master WHERE type='table' AND name='{name}'; """).fetchall()
        if listOfTables == []:
            return False
        else:
            return True

    #create table
    def CreateReplaceTable(self, Data:pd.DataFrame, tablename: str):
        con = sqlite3.connect('CSVs\Stock_data\RawData.db')
        Data.to_sql(tablename, con, if_exists="replace",index=False)

    #Get data by ticker
    def GetDataByTicker(self, ticker:str, name:str) -> pd.DataFrame:
        con = sqlite3.connect('CSVs\Stock_data\RawData.db')
        return  pd.read_sql_query(f"""SELECT * FROM {name} WHERE Ticker='{ticker}'; """, con)
    
    #get all data
    def GetAllData(self, name:str) -> pd.DataFrame:
        con = sqlite3.connect('CSVs\Stock_data\RawData.db')
        return  pd.read_sql_query(f"SELECT * FROM {name}", con)
    

    ###############Analysis Methods#######################
    def AnalysisMethod(self, Ticker:str,TableName:str):
        try:
            dfOrig = self.GetDataByTicker(Ticker,TableName)
            dfOrig['Date'] = pd.to_datetime(dfOrig['Date'])
            dfOrig.set_index('Date', inplace=True)
            dfData = dfOrig.copy()
            if(self.PreFilterStock(dfData)):
                #Moving Averages
                dfData['ma5'] = dfData['Close'].rolling(5).mean()
                dfData['ma10'] = dfData['Close'].rolling(10).mean()
                dfData['ma20'] = dfData['Close'].rolling(20).mean()
                dfData['ma30'] = dfData['Close'].rolling(30).mean()
                dfData['ma40'] = dfData['Close'].rolling(40).mean()
                dfData['ma50'] = dfData['Close'].rolling(50).mean()
                dfData['ma60'] = dfData['Close'].rolling(60).mean()
                dfData['ma70'] = dfData['Close'].rolling(70).mean()

                #Bools for Moving averages
                dfData["5v10"] = dfData.apply(lambda x: MAGreater(x["ma5"],x["ma10"]), axis=1).astype(bool)
                dfData["5v20"] = dfData.apply(lambda x: MAGreater(x["ma5"],x["ma20"]), axis=1).astype(bool)
                dfData["5v30"] = dfData.apply(lambda x: MAGreater(x["ma5"],x["ma30"]), axis=1).astype(bool)
                dfData["5v40"] = dfData.apply(lambda x: MAGreater(x["ma5"],x["ma40"]), axis=1).astype(bool)
                dfData["5v50"] = dfData.apply(lambda x: MAGreater(x["ma5"],x["ma50"]), axis=1).astype(bool)
                dfData["5v60"] = dfData.apply(lambda x: MAGreater(x["ma5"],x["ma60"]), axis=1).astype(bool)
                dfData["5v70"] = dfData.apply(lambda x: MAGreater(x["ma5"],x["ma70"]), axis=1).astype(bool)

                dfData["10v20"] = dfData.apply(lambda x: MAGreater(x["ma10"],x["ma20"]), axis=1).astype(bool)
                dfData["10v30"] = dfData.apply(lambda x: MAGreater(x["ma10"],x["ma30"]), axis=1).astype(bool)
                dfData["10v40"] = dfData.apply(lambda x: MAGreater(x["ma10"],x["ma40"]), axis=1).astype(bool)
                dfData["10v50"] = dfData.apply(lambda x: MAGreater(x["ma10"],x["ma50"]), axis=1).astype(bool)
                dfData["10v60"] = dfData.apply(lambda x: MAGreater(x["ma10"],x["ma60"]), axis=1).astype(bool)
                dfData["10v70"] = dfData.apply(lambda x: MAGreater(x["ma10"],x["ma70"]), axis=1).astype(bool)

                dfData["20v30"] = dfData.apply(lambda x: MAGreater(x["ma20"],x["ma30"]), axis=1).astype(bool)
                dfData["20v40"] = dfData.apply(lambda x: MAGreater(x["ma20"],x["ma40"]), axis=1).astype(bool)
                dfData["20v50"] = dfData.apply(lambda x: MAGreater(x["ma20"],x["ma50"]), axis=1).astype(bool)
                dfData["20v60"] = dfData.apply(lambda x: MAGreater(x["ma20"],x["ma60"]), axis=1).astype(bool)
                dfData["20v70"] = dfData.apply(lambda x: MAGreater(x["ma20"],x["ma70"]), axis=1).astype(bool)

                dfData["30v40"] = dfData.apply(lambda x: MAGreater(x["ma20"],x["ma40"]), axis=1).astype(bool)
                dfData["30v50"] = dfData.apply(lambda x: MAGreater(x["ma20"],x["ma50"]), axis=1).astype(bool)
                dfData["30v60"] = dfData.apply(lambda x: MAGreater(x["ma20"],x["ma60"]), axis=1).astype(bool)
                dfData["30v70"] = dfData.apply(lambda x: MAGreater(x["ma20"],x["ma70"]), axis=1).astype(bool)

                #RSI
                dfData["rsi"] = pta.rsi(dfData['Close'], length = 14)
                dfData["rsiBool"] = dfData.apply(lambda x: RSIgood(x["rsi"],30), axis=1).astype(bool)

                #setting up predicted values
                dfData['Closefuture'] = dfData['Close'].shift(self.DaysAhead)
                dfData["CloseBuy"] = dfData.apply(lambda x: BuyPercent(x["Close"],x["Closefuture"],self.TargetPercentGain), axis=1).astype(bool)
                
                #removing excess data
                #dfData = dfData.iloc[:self.DaysAhead]
                dfData = dfData.iloc[70:] #max moving average

                DfDatavalue, stock_stats = self.MakePredictions(dfData,self.TestSize,Ticker,dfOrig)
                DfDatafin,temp = self.MakePredictions(dfData,self.ActualPredSize,Ticker,dfOrig)
                stock_stats.Gain = DfDatavalue.iloc[-1]['Value']
                stock_stats.price = DfDatavalue.iloc[-1]["Close"]

                plt.close('all')


                if self.PostFilterStock(DfDatavalue,DfDatafin):
                    return stock_stats
                else:
                    return None
            else:
                return None
        except Exception as e:
            plt.close('all')
            print(e)
            return None
        
    def PreFilterStock(self, DfData:pd.DataFrame) -> bool:
        if(DfData.iloc[-1]['Close'] > 10 and DfData.iloc[-1]['Close'] < 300):
            return True
        return False
    
    def PostFilterStock(self, DfData:pd.DataFrame, DFBuy:pd.DataFrame) -> bool:
        df = DFBuy.copy()
        df.drop(df.tail(-self.DaysAhead).index,inplace = True)
        df = df.tail(self.BuySignalsinPast)
        print("last value:" + str(df.iloc[-1]['Value']))
        print("any buy signals:" + str(df['Buy'].any()))

        if(DfData.iloc[-1]['Value'] > self.StartTestMoney and df['Buy'].any()):
            return True
        return False
    
    def MakePredictions(self,dfData:pd.DataFrame, Split:float, Ticker:str,dfOrig:pd.DataFrame):
            train_set: pd.DataFrame
            test_set: pd.DataFrame
            train_set, test_set= train_test_split(dfData, test_size=Split, shuffle=False)
            Y_Train = train_set[self.predFeature]
            X_Train = train_set[self.train_features]

            y_test = test_set[self.predFeature]
            x_test = test_set[self.train_features]

            #X_Train, Y_Train
            model = XGBClassifier(
                tree_method="hist",
                enable_categorical=True,
                n_estimators=800,
                max_leaves=0,
                grow_policy='lossguide',
                )
            
            model.fit(X_Train,Y_Train)

            #set up object for item
            stock_stats = StockStats(Ticker)
            temp1,temp2 = stock_stats.evaluateRegression(model,X_Train,Y_Train,x_test,y_test)
            Predvtest = pd.DataFrame(y_test.copy())
            Predvtest["pred"] = temp2
            stock_stats.ComparePredictions(Predvtest)
            OutFileDF = Predvtest.merge(dfOrig, left_index=True, right_index=True).drop("CloseBuy", axis=1).rename(columns={"pred":"Buy"})

            #get performance stats
            stock_stats.MovingAverages(dfData)
            OutFileDF["Buy"] = OutFileDF["Buy"].astype(bool)
            DfData = OutFileDF.copy()
            DfData = SellPattern(DfData,self.percentsell)
            stock_stats.SellBuyGraphCreation(DfData)
            DfDataOwn = OwnPattern(DfData)
            stock_stats.OwnedGraphCreation(DfDataOwn)
            DfDatavalue = TotalValue(DfDataOwn,self.StartTestMoney)
            stock_stats.ValueGraphCreation(DfDatavalue)
            plt.close('all')
            return DfDatavalue, stock_stats
    
    def PerformAnalysis(self, ListOfTickers:[str], Tname:str) -> str:
            totalstr = ""
            counter = 1
            total = len(ListOfTickers)
            for ticker in ListOfTickers:
                data = self.AnalysisMethod(ticker, Tname)
                if data != None:
                    print(str(counter)+ "/" + str(total) + " Good Stock found: " + ticker)
                    totalstr = totalstr + data.GenerateHTMLPage()
                else:
                    print(str(counter)+ "/" + str(total) + " Bad Stock found: " + ticker)
                counter += 1

            return totalstr
    
    def PerformAnalysisThread(self, ListOfTickers: list[str],retstr:list[str]):
        totalstr = ""
        counter = 1
        total = len(ListOfTickers)
        for ticker in ListOfTickers:
            data = self.AnalysisMethod(ticker)
            if data != None:
                print(str(counter)+ "/" + str(total) + " Good Stock found: " + ticker)
                totalstr = totalstr + data.GenerateHTMLPage()
            else:
                print(str(counter)+ "/" + str(total) + " Bad Stock found: " + ticker)
            counter += 1

        retstr.append(totalstr)
    


    def PerformAnalysisMulti(self, ListOfTickers:list[str]) -> str:
        Tickers1, Tickers2 = split_list(ListOfTickers)
        retstr1 = []
        retstr2 = []
        t1 = threading.Thread(group=None,target=self.PerformAnalysisThread, args=(Tickers1,retstr1,))
        t2 = threading.Thread(group=None,target=self.PerformAnalysisThread, args=(Tickers2,retstr2,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        return retstr1[0] + retstr2[0]
