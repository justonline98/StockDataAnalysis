import time
from Helpers.AutomationClass import StockAutomation as SA
import pandas as pd

start = time.time()

Autoclass = SA()
#print(Autoclass.GetSP500Companies())
#Autoclass.GetStockData(Autoclass.GetNYSECompanies()[0:2])
print("Getting companies for today")
#stocks, name = Autoclass.GetSP400Companies()
stocks, name = Autoclass.GetSP500Companies()
print("Getting data for companies")
Autoclass.GetStockData(stocks,name)
print("Performing analysis on companies")
data = Autoclass.PerformAnalysis(stocks,name)#[11:15]
#data = Autoclass.PerformAnalysisMulti(stocks)
print("Outputing to file test.html")
if data != "":
    print("found good stock")
    text_file = open(name + ".html", "w",encoding='utf-8')
    text_file.write(data)
    text_file.close()
else:
    print("no stocks found for today")

end = time.time()
print(f"Total Time taken: {(end-start)*10**3:.03f}ms")