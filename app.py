import streamlit as st,pandas as pd, numpy as np, yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import date,timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import pandas_datareader as web
import pandas_ta as ta
# https://tradewithpython.com/generating-buy-sell-signals-using-python
st.title('Stock Tracker')
ticker=st.sidebar.text_input('Stock Symbol',value='MSFT')
sd=st.sidebar.date_input('**Start Date**',value=date.today()-timedelta(days=7))
ed=st.sidebar.date_input('**End Date**')
data=yf.download(ticker,start=sd,end=ed)
j=yf.Ticker(ticker) 
# img=px.line(data,x=data.index,y=data['Adj Close'],title='Performance Chart'+" "+j.info['longName']+"("+ ticker+")")
img=px.line(data,x=data.index,y=data['Adj Close'],title='Performance Chart'+" "+ticker)
st.plotly_chart(img)
Data,Statistics,News,Predict,Indicators=st.tabs(["**Price Data**","**Statistics**","**News**","**Predict**","**Indicators**"])
with Data:
    st.write("Data")
    df=data
    df['Change%']=data['Adj Close']/data['Adj Close'].shift(1)-1
    st.write(df)
with Statistics:
    st.header("Statistical Data")
    n=yf.Ticker(ticker)
#     st.write("Market Cap:",n.info['marketCap'])
#     st.write("Trailing P/E:",n.info['trailingPE'])
#     st.write("Forward P/E:",n.info['forwardPE'])
#     st.write("Price/Book:",n.info['priceToBook'])
#     st.write("Price/Sales (ttm):",n.info['priceToSalesTrailing12Months'])

    df=data
    df['Change%']=data['Adj Close']/data['Adj Close'].shift(1)-1
    df.dropna(inplace=True)
    annual_return=df['Change%'].mean()*262*100
    st.write("Annual Return is:",annual_return,'%')
    sdev=np.std(df['Change%']*np.sqrt(252))
    st.write("Standard Deviation is:",sdev*100,'%')
    st.write("Risk Adj. Return is:",annual_return/(sdev*100))
    st.subheader('INCOME STATEMENT')
    n.income_stmt  
    st.subheader('BALANCE SHEET')
    n.balance_sheet
    st.subheader('CASHFLOW STATEMENT')
    n.cashflow
#     # key='OW1639L63B5UCYYL'
#     # fd=FundamentalData(key,output_format='pandas')
#     # st.subheader('Balance Sheet')
#     # bs=fd.get_balance_sheet_annual(ticker)[0]
#     # b=bs.T[2:]
#     # b.columns=list(bs.T.iloc[0])
#     # st.write(b)
#     # st.subheader('Income Statement')
#     # ism=fd.get_income_statement_annual(ticker)[0]
#     # is1=ism.T[2:]
#     # is1.columns=list(ism.T.iloc[0])
#     # st.write(is1)
#     # st.subheader('Cash Flow Statement')
#     # cf=fd.get_cash_flow_annual(ticker)[0]
#     # c=cf.T[2:]
#     # c.columns=list(cf.T.iloc[0])
#     # st.write(c)
from stocknews import StockNews
with News:
    st.write("Top 10 news")
    st.header(f'News of {ticker}')
    amu=[]
    amu.append(ticker)
    sn=StockNews(amu,save_news=False)
    dn=sn.read_rss()
    for i in range(10):
        st.subheader(f'News {i+1}')
        st.write(dn['published'][i])
        st.write(dn['summary'][i])
        st.write(dn['title'][i])
        ts=dn['sentiment_title'][i]
        st.write(f'**Title Sentiment**: {ts}')
        ns=dn['sentiment_summary'][i]
        st.write(f'**News Sentiment:** {ns}')

    
with Predict:
    def model_engine(num):
        d = data[['Adj Close']]
            # shifting the closing price based on number of days forecast
        d['preds'] = data['Adj Close'].shift(-num)
        # scaling the data
        x = d.drop(['preds'], axis=1).values
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        # storing the last num_days data
        x_forecast = x[-num:]
        # selecting the required values for training
        x = x[:-num]
        # getting the preds column
        y = d.preds.values
        # selecting the required values for training
        y = y[:-num]

        #spliting the data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
        # training the model

        r=SVR(kernel='rbf',C=1e3,gamma=0.1)
        r.fit(x_train, y_train)
        preds = r.predict(x_test)
        st.text(f'r2_score: {r2_score(y_test, preds)} \
                \nMAE: {mean_absolute_error(y_test, preds)} \
                \nScore:{r.score(x_test,y_test)}')
        # predicting stock price based on the number of days
        forecast_pred = r.predict(x_forecast)
        day = 1
        for i in forecast_pred:
            st.text(f'Day {day}: {i}')
            day += 1
        
        # X=list(range(1,len(forecast_pred)+1))
        # chart_data = pd.DataFrame(
        # {
        #     "Day": list(range(1,len(forecast_pred)+1)) ,
        #     "Price": forecast_pred})
        # st.line_chart(chart_data, x="Day", y="Price")
        return ""
    c1,c2,c3,c4=st.columns(4)
    with c1:
        if st.button('**1 Days**',type="primary"):
            st.write(model_engine(1))

    with c2:
        if st.button('**5 Days**',type="primary"):
            st.write(model_engine(5))
    with c3:
        if st.button('**7 Days**',type="primary"):
            st.write(model_engine(7))
    with c4:
        if st.button('**30 Days**',type="primary"):
            st.write(model_engine(30))
     
with Indicators:
    st.subheader("**Buy and Sell Using SMA**")
    dj=yf.download(ticker,period='2y',interval='1d')
    # dj=data
    dj['SMA 30'] = ta.sma(dj['Close'],30)
    dj['SMA 100'] = ta.sma(dj['Close'],100)
    #SMA BUY SELL
    #Function for buy and sell signal
    def buy_sell(dj):
        signalBuy = []
        signalSell = []
        position = False 

        for i in range(len(dj)):
            if dj['SMA 30'][i] > dj['SMA 100'][i]:
                if position == False :
                    signalBuy.append(dj['Adj Close'][i])
                    signalSell.append(np.nan)
                    position = True
                else:
                    signalBuy.append(np.nan)
                    signalSell.append(np.nan)
            elif dj['SMA 30'][i] < dj['SMA 100'][i]:
                if position == True:
                    signalBuy.append(np.nan)
                    signalSell.append(dj['Adj Close'][i])
                    position = False
                else:
                    signalBuy.append(np.nan)
                    signalSell.append(np.nan)
            else:
                signalBuy.append(np.nan)
                signalSell.append(np.nan)
        return pd.Series([signalBuy, signalSell])
    dj['Buy_Signal_price'], dj['Sell_Signal_price'] = buy_sell(dj)
    fig, ax = plt.subplots(figsize=(15,9))
    ax.plot(dj['Adj Close'] , label =ticker ,linewidth=2, color='magenta', alpha = 0.9)
    ax.plot(dj['SMA 30'], label = 'SMA30',linewidth=2, alpha = 0.85)
    ax.plot(dj['SMA 100'], label = 'SMA100',linewidth=2, alpha = 0.85)
    marker_size = 200
    ax.scatter(dj.index , dj['Buy_Signal_price'] , label = 'Buy' , marker = '^',s=marker_size, color = 'green',alpha =1 )
    ax.scatter(dj.index , dj['Sell_Signal_price'] , label = 'Sell' , marker = 'v',s=marker_size, color = 'red',alpha =1 )
    ax.set_title(ticker + " Price History with buy and sell signals",fontsize=20, backgroundcolor='cyan', color='black')
    ax.set_xlabel(f'{sd} - {ed}' ,fontsize=18)
    ax.set_ylabel('Close Price' , fontsize=18)
    legend = ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)


    st.subheader("**MACD**")
    macd = ta.macd(dj['Close'])
    dj = pd.concat([dj, macd], axis=1).reindex(dj.index)
    def MACD_Strategy(df, risk):
        MACD_Buy=[]
        MACD_Sell=[]
        position=False

        for i in range(0, len(df)):
            if df['MACD_12_26_9'][i] > df['MACDs_12_26_9'][i] :
                MACD_Sell.append(np.nan)
                if position ==False:
                    MACD_Buy.append(df['Adj Close'][i])
                    position=True
                else:
                    MACD_Buy.append(np.nan)
            elif df['MACD_12_26_9'][i] < df['MACDs_12_26_9'][i] :
                MACD_Buy.append(np.nan)
                if position == True:
                    MACD_Sell.append(df['Adj Close'][i])
                    position=False
                else:
                    MACD_Sell.append(np.nan)
            elif position == True and df['Adj Close'][i] < MACD_Buy[-1] * (1 - risk):
                MACD_Sell.append(df["Adj Close"][i])
                MACD_Buy.append(np.nan)
                position = False
            elif position == True and df['Adj Close'][i] < df['Adj Close'][i - 1] * (1 - risk):
                MACD_Sell.append(df["Adj Close"][i])
                MACD_Buy.append(np.nan)
                position = False
            else:
                MACD_Buy.append(np.nan)
                MACD_Sell.append(np.nan)

        dj['MACD_Buy_Signal_price'] = MACD_Buy
        dj['MACD_Sell_Signal_price'] = MACD_Sell

    MACD_strategy = MACD_Strategy(dj, 0.025)
    
    def MACD_color(data):
        MACD_color = []
        for i in range(0, len(data)):
            if data['MACDh_12_26_9'][i] > data['MACDh_12_26_9'][i - 1]:
                MACD_color.append(True)
            else:
                MACD_color.append(False)
        return MACD_color
    
    dj['positive'] = MACD_color(dj)
    plt.rcParams.update({'font.size': 10})
    fig, ax1 = plt.subplots(figsize=(13,9))
    fig.suptitle(ticker, fontsize=20, backgroundcolor='cyan', color='black')
    ax1 = plt.subplot2grid((14, 8), (0, 0), rowspan=8, colspan=14)
    ax2 = plt.subplot2grid((14, 12), (10, 0), rowspan=6, colspan=14)
    ax1.set_ylabel('Price',fontsize=15)
    ax1.plot('Adj Close',data=dj, label='Close Price', linewidth=1.5, color='blue')
    ax1.scatter(dj.index, dj['MACD_Buy_Signal_price'], color='green', marker='^',s=marker_size, alpha=1)
    ax1.scatter(dj.index, dj['MACD_Sell_Signal_price'], color='red', marker='v',s=marker_size, alpha=1)
    ax1.legend()
    ax1.grid()
    ax1.set_xlabel('Date', fontsize=15)

    ax2.set_ylabel('MACD', fontsize=15)
    ax2.plot('MACD_12_26_9', data=dj, label='MACD', linewidth=1, color='blue')
    ax2.plot('MACDs_12_26_9', data=dj, label='signal', linewidth=1, color='red')
    ax2.bar(dj.index,'MACDh_12_26_9', data=dj, label='Volume', color=dj.positive.map({True: 'g', False: 'r'}),width=3,alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax2.grid()
    plt.show()
    st.pyplot(plt)


    

   
    
    
    

