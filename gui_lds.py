from asyncio import futures
from calendar import day_abbr
from math import sqrt
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from PIL import Image
import altair as alt
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import time 
from sklearn.model_selection import train_test_split
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
#Import data
data=pd.read_csv('avocado.csv')

st.title("Data Science Project 1 ")
st.subheader("Hass avocado price prediction")

#upload file
upload_file=st.file_uploader("Choose a file", type=['csv'])
if upload_file is not None:
   data=pd.read_csv(upload_file)
   data.to_csv('avocado_new.csv', index=False)
#rename columns
data.rename(columns = {'Total Volume':'TotalVolumne','Total Bags': 'TotalBags', 'Small Bags': 'SmallBags', 'Large Bags':'LargeBags', 'XLarge Bags': 'XLargeBags'}, inplace = True)
##Tạo cột tháng, quý từ dữ liệu Date
data['Date'] = pd.to_datetime(data['Date'])
data['months'] = pd.DatetimeIndex(data['Date']).month
data['month_year'] = data['Date'].dt.strftime('%Y-%m')
data['quarter'] = data['Date'].dt.to_period('Q')


#build model random forest
input = data[['TotalVolumne','4046','4225','4770','TotalBags','SmallBags','LargeBags','type']]
output = data[['AveragePrice']]
##Robust scaler
scaler = RobustScaler()
df_scaler = scaler.fit_transform(input[['TotalVolumne','4046','4225','4770','TotalBags','SmallBags','LargeBags']])
df_scaler = pd.DataFrame(df_scaler,columns=['TotalVolumne','4046','4225','4770','TotalBags','SmallBags','LargeBags'])
##Lable Encoder
encoder = LabelEncoder()
input['type_encoder'] = encoder.fit_transform(input['type'])
df_type = input['type_encoder']
##Determine X, y
X = pd.concat([df_scaler,df_type],axis=1)
y = data['AveragePrice']
##select best model
models = [
    LinearRegression(),
    KNeighborsRegressor(),
    DecisionTreeRegressor(random_state=12),
    RandomForestRegressor(n_estimators=30),
    SVR(kernel='rbf')
]
df_model = pd.DataFrame()
entries = []
for model in models:
    start = time.time()
    model.fit(X,y)
    end = time.time()
    duration = end - start
    model_name = model.__class__.__name__ 
    R_square = round(model.score(X,y),3)
    yhat = model.predict(X)
    RMSE = round(mean_squared_error(y,yhat,squared=False),3)
    entries.append([model_name,R_square,RMSE,duration])
df_model = pd.DataFrame(entries, columns = ['model_name','R_square','RMSE','duration'])
df_model = df_model.sort_values(by='R_square',ascending=False)
#select train and test set
lst_test_size = [0.3,0.25,0.2]
entries_rate = []
train_test_rate = pd.DataFrame()
for i in lst_test_size:
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = i, random_state = 12)
  model = RandomForestRegressor(n_estimators=30)
  model.fit(X_train,y_train)
  test_size = i
  train_size = 1 - i
  yhat_train = model.predict(X_train)
  yhat_test = model.predict(X_test)
  R_square_train = round(model.score(X_train,y_train),3)
  R_square_test = round(model.score(X_test,y_test),3)
  entries_rate.append([test_size,train_size,R_square_train,R_square_test])
train_test_rate = pd.DataFrame(entries_rate, columns = ['test_size','train_size','R_square_train','R_square_test'])
train_test_rate = train_test_rate.sort_values(by='R_square_train',ascending=False)
##Build Random forest model
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25,random_state=12)
model_rf = RandomForestRegressor(n_estimators=30)
model_rf.fit(X_train,y_train)
yhat_train = model.predict(X_train)
yhat_test = model.predict(X_test)
R_square_train = round(model_rf.score(X_train,y_train),3)
R_square_test = round(model_rf.score(X_test,y_test),3)
RMSE = round(mean_squared_error(y_test,yhat_test,squared=False),3)

#time series model
df_conven = data[data['type']=='conventional']
df_organic = data[data['type']=='organic']
cali_cv = df_conven[df_conven['region']=='California']
west_cv = df_conven[df_conven['region']=='West']
north_cv = df_conven[df_conven['region']=='Northeast']
north_or = df_organic[df_organic['region']=='Northeast']
cali_or = df_organic[df_organic['region']=='California']
##prepare data for cali conventional avocado
cali_cv = cali_cv[['Date','AveragePrice']]
cali_cv.sort_values(by = ['Date'], ascending = True, inplace = True)
cali_cv['date'] = cali_cv['Date'].to_numpy().astype('datetime64[W]')
cali_cv = cali_cv.groupby(by = ['date'])['AveragePrice'].mean().reset_index()
cali_cv.set_index('date', inplace = True)
cali_cv1 = cali_cv.groupby(by = cali_cv.index)['AveragePrice'].mean().reset_index()
cali_cv1.rename(columns = {'date':'ds','AveragePrice':'y'}, inplace = True)
cali_cv_fb_model = Prophet(interval_width=0.95,
                           weekly_seasonality=True,
                           daily_seasonality=True,
                           yearly_seasonality=True)
cali_cv_fb_model.fit(cali_cv1)
weeks = pd.date_range('2018-01-01','2018-12-01',
                       freq = 'W').strftime('%Y-%m-%d').tolist()

future_1 = pd.DataFrame(weeks) 
future_1.columns = ['ds']
future_1['ds'] = pd.to_datetime(future_1['ds'])

forecast_1 = cali_cv_fb_model.predict(future_1)
#calculate MAE /RMSE between actual and predicted
y_test1=cali_cv['AveragePrice'].values[:10]
y_pred=forecast_1['yhat'].values[:10]
mae_p=mean_absolute_error(y_test1,y_pred)
rmse_p=sqrt(mean_absolute_error(y_test1,y_pred))


###long term prediction for the next 5 years
m_1 = Prophet() 
m_1.fit(cali_cv1)
future_5y1 = m_1.make_future_dataframe(periods=48*5, freq='W')
forecast_cali_cv = m_1.predict(future_5y1)



##prepare data for cali organic avocado
cali_or = cali_or[['Date','AveragePrice']]
cali_or.sort_values(by = ['Date'], ascending = True, inplace = True)

cali_or['date'] = cali_or['Date'].to_numpy().astype('datetime64[W]')
cali_or = cali_or.groupby(by = ['date'])['AveragePrice'].mean().reset_index()

cali_or.set_index('date', inplace = True)
cali_or1= cali_or.groupby(by = cali_or.index)['AveragePrice'].mean().reset_index()
cali_or1.rename(columns = {'date':'ds','AveragePrice':'y'}, inplace = True)
#Buil Model
cali_or_fb_model = Prophet(interval_width=0.95,
                           daily_seasonality=True,
                           weekly_seasonality=True,
                           yearly_seasonality=True)
#Fit model
cali_or_fb_model.fit(cali_or1)
weeks = pd.date_range('2018-01-01','2018-12-01',
                       freq = 'W').strftime('%Y-%m-%d').tolist()

future_2 = pd.DataFrame(weeks) 
future_2.columns = ['ds']
future_2['ds'] = pd.to_datetime(future_2['ds'])

forecast_2 = cali_or_fb_model.predict(future_2)
y_test2=cali_or['AveragePrice'].values[:10]
y_pred2=forecast_2['yhat'].values[:10]
mae_p2=mean_absolute_error(y_test2,y_pred2)
rmse_p2=sqrt(mean_absolute_error(y_test2,y_pred2))

###long term prediction for the next 5 years
m_2 = Prophet() 
m_2.fit(cali_or1)
future_5y2 = m_2.make_future_dataframe(periods=48*5, freq='W')
forecast_5y2 = m_2.predict(future_5y2)
##prepare data for west conventional avocado
west_cv = west_cv[['Date','AveragePrice']]
west_cv.sort_values(by = ['Date'], ascending = True, inplace = True)

west_cv['date'] = west_cv['Date'].to_numpy().astype('datetime64[W]')
west_cv = west_cv.groupby(by = ['date'])['AveragePrice'].mean().reset_index()

west_cv.set_index('date', inplace = True)
west_cv1 = west_cv.groupby(by = west_cv.index)['AveragePrice'].mean().reset_index()
west_cv1.rename(columns = {'date':'ds','AveragePrice':'y'}, inplace = True)

#Buil Model
west_cv_fb_model = Prophet(interval_width=0.95,
                           daily_seasonality=True,
                           weekly_seasonality=True,
                           yearly_seasonality=True)

#Fit model
west_cv_fb_model.fit(west_cv1)
weeks = pd.date_range('2018-01-01','2018-12-01',
                       freq = 'W').strftime('%Y-%m-%d').tolist()

future_3 = pd.DataFrame(weeks) 
future_3.columns = ['ds']
future_3['ds'] = pd.to_datetime(future_3['ds'])

forecast_west = west_cv_fb_model.predict(future_3)
y_test3=west_cv['AveragePrice'].values[:10]
y_pred3=forecast_west['yhat'].values[:10]
mae_p3=mean_absolute_error(y_test3,y_pred3)
rmse_p3=sqrt(mean_absolute_error(y_test3,y_pred3))

#Long-term prediction for the next 5 years
m_3 = Prophet() 
m_3.fit(west_cv1)
future_5y3 = m_3.make_future_dataframe(periods=48*5, freq='W')
forecast_3 = m_3.predict(future_5y3)

##prepare data for north organic avocado
north_or = north_or[['Date','AveragePrice']]
north_or.sort_values(by = ['Date'], ascending = True, inplace = True)

north_or['date'] = north_or['Date'].to_numpy().astype('datetime64[W]')
north_or = north_or.groupby(by = ['date'])['AveragePrice'].mean().reset_index()
north_or.set_index('date', inplace = True)

north_or1 = north_or.groupby(by = north_or.index)['AveragePrice'].mean().reset_index()
north_or1.rename(columns = {'date':'ds','AveragePrice':'y'}, inplace = True)
north_or_fb_model = Prophet()
#Fit model
north_or_fb_model.fit(north_or1)
weeks = pd.date_range('2018-01-01','2018-12-01',
                       freq = 'W').strftime('%Y-%m-%d').tolist()

future4 = pd.DataFrame(weeks) 
future4.columns = ['ds']
future4['ds'] = pd.to_datetime(future4['ds'])

forecast_north = north_or_fb_model.predict(future4)
y_test4=north_or['AveragePrice'].values[:10]
y_pred4=forecast_north['yhat'].values[:10]
mae_p4=mean_absolute_error(y_test4,y_pred4)
rmse_p4=sqrt(mean_absolute_error(y_test4,y_pred4))
#Long-term prediction for the next 5 years
m_4 = Prophet(yearly_seasonality=False, \
            daily_seasonality=False, weekly_seasonality=True) 
m_4.fit(north_or1)
future_5y4 = m_4.make_future_dataframe(periods=48*5, freq='W')
forecast_4 = m_4.predict(future_5y4)
##__________________________________
###GUI###
menu = ["Business Objective","Data Overview", "Build model",'Show prediction','Business strategy']
choice = st.sidebar.selectbox('Menu', menu)
if choice =='Business Objective':
    st.subheader('Business Objective')
    st.write("""
    #####Bơ “Hass", một công ty có trụ sở tại Mexico,chuyên sản xuất nhiều loại quả bơ được bán ở Mỹ.Họ đã rất thành công trong những năm gần đây và muốn mở rộng. Vì vậy, họ muốn xây dựng mô hình hợp lý để dự đoán giá trung bình của bơ “Hass” ở Mỹ nhằm xem xét việc mở rộng các loại trang trại Bơ đang có cho việc trồng bơ ở các vùng khác.#####
    """)
    image = Image.open('D:\\images.png')
    st.image(image)
    st.write("""
    ####Goal: Xây dựng mô hình dự đoán giá bơ hass =>Xem xét việc mở rộng và phát triển kinh doanh #####
    """)
    st.write("""
    ###Part 1: Analize Business Overview""")
    st.write("""
    ###Part 2: Build random forest model to predict prict""")
    st.write("""
    ###Part 3: Build time-series model with specific region (fbprophet and arima)""")
elif choice=='Build model':
    st.subheader('***Build model***')
    st.subheader('Bài toán 1: USA’s Avocado AveragePrice Prediction – Sử dụng các thuật toán Regression như Linear Regression, Random Forest, XGB Regressor...')
    st.subheader("""Data Transformation""")
    st.subheader("""Lựa chọn biến Input:
    Continous variable: TotalVolumne, 4046, 4225, 4770, TotalBags, SmallBags, LargeBags.
    Category variable: type.""")
    st.subheader("""Biến Output: AveragePrice""")
    st.write('Sử dụng Robust Scaler cho các biến continuous trong input')
    st.caption('Các biến sau khi scale:')
    st.dataframe(df_scaler)
    st.write('Chọn lựa best model để đem lại kết quả tốt nhất')
    st.caption('Kết quả các thuật toán')
    st.dataframe(df_model)
    st.write('Từ các thông số R_square, RMSE và duration của các model, lựa chọn RandomForestRegressor cho tập dữ liệu.')
    st.write("""####Select train and test rate###
             """)
    st.caption('train and test efficiency')
    st.dataframe(train_test_rate)
    st.write('Từ kết quả trên lựa chọn tỷ lệ test_size = 0.25 và train_size = 0.75 cho RandomForestRegressor model.')
    st.caption('đánh giá kết quả mô hình')
    st.code("R square of train: "+str(round(R_square_train,2)))
    st.code("R square of test: "+str(round(R_square_test,2)))
    st.code("RMSE: "+str(round(RMSE,2)))
    st.caption('kde plot of Model')
    fig13=plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    sns.kdeplot(y_train, color='blue', label='True train values')
    sns.kdeplot(yhat_train, color='red', label='Predict train values')
    plt.title('y_train vs yhat_train')
    plt.subplot(1,2,2)
    sns.kdeplot(y_test, color='blue', label='True test values')
    sns.kdeplot(yhat_test, color='red', label='Predict test values')
    plt.title('y_test vs yhat_test')
    plt.legend()
    st.pyplot(fig13)
    st.caption('Visualize correlation Actual Value & Predict Value')
    fig14=plt.figure(figsize=(14,7))
    plt.scatter(yhat_test, y_test)
    plt.xlabel('Model predictions')
    plt.ylabel('True values')
    plt.plot([0, 3], [0, 3], 'r-')
    plt.title('Evaluate Actual Value & Predict Value')
    st.pyplot(fig14)
    st.subheader('Bài toán 2 Conventional/Organic Avocado Average Price Prediction for the future in California/NewYork... - sử dụng các thuật toán Time Series như ARIMA, Prophet...')
    st.write('Từ kết quả phân tích trong bài toán 1, lựa chọn 3 vùng California, West và Northeast để dự đoán giá trong tương lai, trong đó:')
    st.write('Bơ thường: California, West và Northeast')
    st.write('Bơ hữu cơ: Northeast')
    st.write('Lựa chọn model để đánh giá: Facebook Prophet')
    st.caption('Conventional California')
    fig15=plt.figure(figsize = (10,5))
    plt.plot(cali_cv)
    plt.title('Average Price of Conventional in California from 1/2015 - 3/2018')
    st.pyplot(fig15)
    st.caption('Organic California')
    fig16=plt.figure(figsize = (10,5))
    plt.plot(cali_or)
    plt.title('Average Price of Organic in California from 1/2015 - 3/2018')
    st.pyplot(fig16)
    st.caption('Conventional west')
    fig17=plt.figure(figsize = (10,5))
    plt.plot(west_cv)
    plt.title('Average Price of Conventional in West from 1/2015 - 3/2018')
    st.pyplot(fig17)
    st.caption('Organic north')
    fig18=plt.figure(figsize = (10,5))
    plt.plot(north_or)
    plt.title('Average Price of organic in north from 1/2015 - 3/2018')
    st.pyplot(fig18)
    st.write("""
    ##### Calculate RMSE and MSE between expected and predicted value conventional California""")
    st.code("MAE: "+str(round(mae_p,2)))
    st.code("RMSE: "+str(round(rmse_p,2)))
    st.write("""This results show that model is good enough to predict the conventional avocado average price in California
    """)
    st.write("""
    ##### Calculate RMSE and MSE between expected and predicted value organic avocado average price in California""")
    st.code("MAE: "+str(round(mae_p2,2)))
    st.code("RMSE: "+str(round(rmse_p2,2)))
    st.write("""This results show that model is good enough to predict the organic avocado average price in California
    """)
    st.write("""
    ##### Calculate RMSE and MSE between expected and predicted value conventional avocado average price in west""")
    st.code("MAE: "+str(round(mae_p3,2)))
    st.code("RMSE: "+str(round(rmse_p3,2)))
    st.write("""This results show that model is good enough to predict the organic avocado average price in west
    """)
    st.write("""
    ##### Calculate RMSE and MSE between expected and predicted value organic avocado average price in north""")
    st.code("MAE: "+str(round(mae_p4,2)))
    st.code("RMSE: "+str(round(rmse_p4,2)))
    st.write("""This results show that model is good enough to predict the organic avocado average price in north
    """)     
elif choice=='Show prediction':
    st.subheader('Predict results of RandomForest Regressor model and actual value')
    df_rf=pd.DataFrame()
    df_rf['y_actual'] = y_test
    df_rf['y_predict'] = yhat_test
    st.dataframe(df_rf)
    st.subheader('Prediction of time series model')
    st.write('###conventional California avocado###')
    ###Show results of prophet model
    st.caption('Prophet models results of convetional California')
    fig = cali_cv_fb_model.plot(forecast_1)
    a = add_changepoints_to_plot(fig.gca(), cali_cv_fb_model, forecast_1)
    st.pyplot(fig)
    fig1 = cali_cv_fb_model.plot_components(forecast_1)
    st.pyplot(fig1)
    st.write('Giá trung bình của bơ thường có xu hướng tăng trong 12 tháng tới.')
    st.write('Đánh giá theo tuần giá bơ sẽ tăng cao vào thứ 5 hàng tuần.')
    st.write('Đánh giá theo năm, giá bơ bắt đầu tăng vào tháng 7 đến tháng 11 của mỗi năm, giảm sâu vào tháng 1.')
    fig2 = m_1.plot(forecast_cali_cv)
    plt.title('Average Price Prediction of Conventional Avocado at California (5 years)')
    a = add_changepoints_to_plot(fig2.gca(), m_1, forecast_cali_cv)
    st.pyplot(fig2)
    st.write('Trong 5 năm tới mô hình dự đoán giá bơ Conventional tại California có xu hướng tăng.')
    
    st.caption('Prophet models results of organic avocado in  California')
    fig3 = cali_or_fb_model.plot(forecast_2)
    a = add_changepoints_to_plot(fig3.gca(), cali_or_fb_model, forecast_2)
    st.pyplot(fig3)
    fig4 = cali_or_fb_model.plot_components(forecast_2)
    st.pyplot(fig4)
    st.write('Giá trung bình của bơ thường có xu hướng tăng trong 12 tháng tới.')
    st.write('Đánh giá theo tuần giá bơ sẽ tăng cao vào thứ 5 hàng tuần.')
    st.write('Đánh giá theo năm, giá bơ bắt đầu tăng vào tháng 7 đến tháng 11 của mỗi năm, giảm sâu vào tháng 1.')
    fig_m = m_2.plot(forecast_5y2)
    plt.title('Average Price Prediction of organic Avocado at California (5 years)')
    a = add_changepoints_to_plot(fig_m.gca(), m_2, forecast_5y2)
    st.pyplot(fig_m)
    st.write('Trong 5 năm tới mô hình dự đoán giá bơ organic tại California có xu hướng tăng.')

    st.caption('Prophet models results of conventional avocado in  west')
    fig20 = west_cv_fb_model.plot(forecast_west)
    a = add_changepoints_to_plot(fig20.gca(), west_cv_fb_model, forecast_west)
    st.pyplot(fig20)
    fig44 = west_cv_fb_model.plot_components(forecast_west)
    st.pyplot(fig44)
    st.write('Giá trung bình của bơ thường có xu hướng tăng trong 12 tháng tới.')
    st.write('Đánh giá theo tuần giá bơ sẽ tăng cao vào thứ 5 hàng tuần.')
    st.write('Đánh giá theo năm, giá bơ bắt đầu tăng vào tháng 7 đến tháng 11 của mỗi năm, giảm sâu vào tháng 1.')
    fig_mm = m_3.plot(forecast_3)
    plt.title('Average Price Prediction of organic Avocado at California (5 years)')
    a = add_changepoints_to_plot(fig_mm.gca(), m_2, forecast_3)
    st.pyplot(fig_mm)
    st.write('Trong 5 năm tới mô hình dự đoán giá bơ conventional tại west có xu hướng tăng.')

    st.caption('Prophet models results of organic avocado in  north')
    fig22 = north_or_fb_model.plot(forecast_north)
    a = add_changepoints_to_plot(fig22.gca(),north_or_fb_model, forecast_north)
    st.pyplot(fig22)
    fig33 = north_or_fb_model.plot_components(forecast_north)
    st.pyplot(fig33)
    st.write('Giá trung bình của bơ thường có xu hướng tăng trong 12 tháng tới.')
    st.write('Đánh giá theo tuần giá bơ sẽ tăng cao vào thứ 5 hàng tuần.')
    st.write('Đánh giá theo năm, giá bơ bắt đầu tăng vào tháng 7 đến tháng 11 của mỗi năm, giảm sâu vào tháng 1.')
    fig_mm1 = m_4.plot(forecast_4)
    plt.title('Average Price Prediction of organic Avocado at California (5 years)')
    a = add_changepoints_to_plot(fig_mm1.gca(), m_4, forecast_4)
    st.pyplot(fig_mm1)
    st.write('Trong 5 năm tới mô hình dự đoán giá bơ organicl tại north có xu hướng giảm.')




elif choice=='Data Overview':
    st.subheader("___________BUSINESS OVERVIEW______________")
    pr=data.profile_report()
    st_profile_report(pr)
    st.write("""#####Từ kết quả thống kê sản lượng mỗi loại bơ theo năm cho thấy, sản lượng bơ thường và bơ hữu cơ không có sự chênh lệch nhiều trong năm.""")
    st.write("""Từ năm 2015 đến 2017 tổng sản lượng bơ của cả 2 loại cao hơn so với sản lượng năm 2018.""")
    st.write("""Vì dữ liệu được ghi nhận từ tháng 1/2015 đến tháng 3/2018, do đó lượng dữ liệu 2018 nhỏ hơn so với 3 năm trước.""")
    st.write("""
    ==> Ảnh hưởng đến việc xác định tập train, test khi xây dựng model.
    """)
    st.subheader("Thống kê sản lượng bơ:")
    table = pd.crosstab(data['year'], data['type'])
    st.dataframe(table)
    st.subheader("Correlation of continuous variables")
    table_corr=data[['TotalVolumne', '4046','4225','4770','TotalBags','SmallBags','LargeBags']].corr()
    st.dataframe(table_corr)
    #region
    region_averageprice = data.groupby(by = ['region','type'])['AveragePrice'].mean().reset_index()
    region_averageprice.sort_values(by = 'AveragePrice', ascending = False, inplace = True)
    region_conven = region_averageprice[region_averageprice['type']=='conventional']
    region_organic = region_averageprice[region_averageprice['type']=='organic']
    #create bar plot of conventional avocado by region
    st.subheader('bar plot of conventional avocado by region')
    fig5 = plt.figure(figsize = (14, 7))
    plt.bar(region_conven['region'], region_conven['AveragePrice'])
    plt.xlabel("Region")
    plt.xticks(rotation = 90)
    plt.ylabel("Average Price")
    plt.title("Average Price of conventional avocado by region")
    st.pyplot(fig5)
    #create bar plot of organic avocado by region
    st.subheader('bar plot of organic avocado by region')
    fig6 = plt.figure(figsize = (14, 7))
    plt.bar(region_organic['region'], region_organic['AveragePrice'])
    plt.xlabel("Region")
    plt.xticks(rotation = 90)
    plt.ylabel("Average Price")
    plt.title("Average Price of organic avocado by region")
    st.pyplot(fig6)
    st.write("""Từ kết quả thống kê trung bình giá bơ của từng loại theo vùng cho thấy, giá bơ có sự khác nhau theo vùng từ 1/2015 đến 3/2018. Bên cạnh đó, giá từng loại bơ khác nhau theo từng vùng.
    """)
    st.write("""####Danh sách 5 vùng có giá bơ trung cao nhất theo loại bơ:(Sắp xếp theo thừ tự giảm dần theo average price""")
    st.write("""Bơ thông thường : HartfordSpringfield, NewYork, SanFrancisco, Philadelphia và Syracuse.""")
    st.write("""Bơ hữu cơ : HartfordSpringfield, SanFrancisco, NewYork, Sacramento, Charlotte.""")
    st.write("""=> Từ kết quả trên cho thấy giá bơ của 2 loại ở 3 vùng HartfordSpringfield, SanFrancisco và Newyork luôn cao hơn so với các vùng khác, đặc biệt vùng HartfordSpringfield cao nhất ở cả 2 loại bơ.
    """)
    st.write("""Kết luận: dựa vào trung bình giá bơ (Average Price) của từng loại theo từng vùng khác nhau, giá trị được ghi nhận từ các nhà bán lẻ của Hass từ tháng 1/2015 đến 3/2018, lựa chọn các vùng HartfordSpringfield, SanFrancisco và Newyork để tập trung vào phát triển mở rộng sản xuất ở cả 2 loại bơ
    """)
    #price by quarter
    st.subheader('price by quarter')
    q_averageprice = data.groupby(by = ['quarter'])['AveragePrice'].mean()
    fig7=plt.figure(figsize = (14, 7))
    q_averageprice.plot(grid = True, figsize = (14,7), title = 'Average Price by Quarter in Year')
    plt.ylabel('Average Price')
    st.pyplot(fig7)
    st.write("""
    Từ biểu đồ line chart của Average theo quý trong năm cho thấy từ năm 2015 đến 2017 giá bơ có xu hướng tăng, đặc biệt tăng mạnh trong năm quý 3 năm 2017. Bên cạnh đó, giá bơ có xu hướng thay đổi theo mùa khi giá bơ giảm mạnh vào quý 1 hàng năm và bắt đầu tăng vào quý 2, tăng mạnh và đạt đỉnh vào quý 3.Mặt khác, giá bơ trong quý 1 của năm 2018 giảm mạnh.
    """)
    #price by month
    st.subheader('price by month')
    m_averageprice = data.groupby(by = ['month_year'])['AveragePrice'].mean()
    fig8=plt.figure(figsize=(14,7))
    m_averageprice.plot(grid = True, figsize = (14,7), title = 'Average Price by Month in Year')
    plt.ylabel('Average Price')
    plt.show()
    st.pyplot(fig8)
    st.write("""Từ kết quả biểu đồ phân tích Average Price theo tháng trong năm cho thấy giá bơ trung bình giảm mạnh từ tháng 8/2017 và giá bơ trong 3 tháng đầu năm 2018 có giá thấp hơn giá trung bình là 1.405.
    """)
    st.write("""Từ dữ liệu ghi nhận từ 1/2015 đến 3/2018, giá bơ co xu hướng giảm trong các tháng cuối năm và bắt đầu tăng từ tháng 7 đến tháng 9 trong năm.
    """)
    #Time series & Type
    st.subheader('price by time-series& type')
    type_averageprice = data.groupby(by = ['type','quarter'])['AveragePrice'].mean().reset_index()
    type_averageprice['quarter'] = type_averageprice['quarter'].astype(str)
    fig9=plt.figure(figsize=(14,7))
    sns.lineplot(x = type_averageprice['quarter'], y = type_averageprice['AveragePrice'], hue = type_averageprice['type']).set_title('Average Price by quater & type')
    st.pyplot(fig9)
    st.write("""Từ kết quả biểu đồ cho thấy giá bơ biến động các quý trong năm không có sự khác biệt giữa 2 loại bơ thường và bơ hữu cơ.""")
    st.write("""Giá bơ hữu cơ cao hơn rất nhiều so với giá bơ thường. Giá bơ của hai loại bơ có xu hướng tăng qua từng năm. Mặt khác, giá bơ của 2 loại đều giảm mạnh đầu năm 2018.Từ kết quả trên, dự đoán ban đầu về việc phát triển mở rộng sản xuất cả 2 loại bơ song song trong thời gian tới.
    """)
    #Total Volumne by type
    st.subheader('Total volumne by type')
    type_volumne = data.groupby(by = ['type','quarter'])['TotalVolumne'].sum().reset_index()
    type_volumne['quarter'] = type_volumne['quarter'].astype(str)
    fig10=plt.figure(figsize = (14,7))
    sns.barplot(x = type_volumne['quarter'], y = type_volumne['TotalVolumne'], hue = type_volumne['type'])
    plt.title('Total Volumne by type & quater in year')
    st.pyplot(fig10)
    st.write("""Từ kết quả thống kê số lượng bơ đã bán của các qúy trong năm theo 2 loại cho thấy, số lượng bơ hưu cơ đã bán thấp hơn nhiều so với bơ thường. Bên cạnh đó, số lượng bơ đã bán trong quý 1 năm 2018 cao hơn so với các quý khác trong các năm trước, tuy nhiêu về thống kê Average Price thì quý 1 năm 2018 tương đối thấp.""")
    #Total Volumne by region
    st.subheader('Total volumne by region')
    region_volumne = data.groupby(by = ['region'])['TotalVolumne'].sum().reset_index()
    region_volumne.sort_values(by = 'TotalVolumne', ascending = False, inplace = True)
    region_vol = region_volumne[region_volumne['region']!='TotalUS']
    fig11=plt.figure(figsize = (14,5))
    sns.barplot(x = region_vol['region'], y = region_vol['TotalVolumne'])
    plt.title('Total Volumne by region (-TotalUS)')
    plt.xticks(rotation = 90)
    st.pyplot(fig11)
    st.write("""Đánh giá theo sản lượng đã tiêu thụ ở 3 vùng HartfordSpringfield, SanFrancisco và Newyork thấp hơn nhiều so với top 10 vùng có sản lượng tiêu thụ cao. Có thể giá trùng bình của 3 vùng này cao ảnh hưởng đến số lượng bơ tiêu thụ của người dùng.""")
    #Sales
    st.subheader("Sales by region")
    data['sales'] = data['TotalVolumne']*data['AveragePrice']
    region_sales = data.groupby(by = ['region'])['sales'].mean().reset_index()
    region_sales.sort_values(by = 'sales', ascending = False, inplace = True)
    region_sales = region_sales[region_sales['region']!='TotalUS']
    fig12=plt.figure(figsize = (14,7))
    sns.barplot(x = 'region', y = 'sales', data = region_sales).set_title('Sales of by Region (1/2015 to 3/2018)')
    plt.xticks(rotation=90)
    st.pyplot(fig12)
    st.write("""Doanh thu bán hàng của vùng California, West, Northeast cáo hơn nhiều so với các vùng còn lại, vì số lượng bơ bán ở 3 vùng này chiếm tỷ trọng cao. Trong khi đó, các vùng có average price cao thì doanh thu tương đối thấp. Vì vậy có thể dự đoán hiện số lượng bơ bán của vùng ảnh hưởng trực tiếp đến doanh thu. Yếu tố sản lượng có thể ảnh hưởng đến quyết định lựa chọn vùng phát triển mở rộng sản lượng trong tương lai.""")
    st.write("""Kết luận: dự vào doanh thu bán hàng từ tháng 1/2015 đến tháng 3/2018, lựa chọn 3 vùng California, West, Northeast cho chiến lược phát triển mở rộng trang trại và sản lượng bơ.""")
    #organic sales
    st.subheader("""Organic Sales""")
    organic = data[data['type']=='organic']
    organic['year'] = organic['year'].astype(str)
    organic = organic[organic['region']!='TotalUS']
    organic_sales = organic.groupby(by = ['region','year'])['sales'].mean().reset_index()
    organic_sales = organic_sales.pivot( index = ['region'],columns=['year'], values = ['sales'])
    st.dataframe(organic_sales)
    st.write('Từ kết quả thống kê doanh thu bơ hữu cơ theo từng vùng và năm cho thấy, doanh thu bơ organic cũng tập trung vào 3 vùng California, Northeast và West.')
    organic_sales_per = round(organic_sales.div(organic_sales.sum(axis=1), axis=0)*100,2)
    organic_sales_per = round(organic_sales.div(organic_sales.sum(axis=1), axis=0)*100,2)
    st.dataframe(organic_sales_per)
    st.write("""Đánh giá mực độ tăng trưởng doanh thu bơ hữu cơ theo từng vùng qua các năm cho thấy, một số vùng có mức tăng trương nhanh về doanh thu bắt đầu từ năm 2017 - tăng trưởng trung bình > 12% so với năm 2015 và 2016.""")
    st.write("""Đánh giá 3 vùng có doanh thu bơ cao nhất (cả 2 loại) là California, West và Northeast thì Northeast có mức tăng trưởng vượt bật từ năm 2017, cụ thể:""")
    st.write("Northeast: 2015 - 8.21, 2016 - 15.83, 2017 - 32.60, 2018(quý 1) - 43.35""")
    st.write("""West: 2015 - 22.50, 2016 - 21.88, 2017 - 25.95, 2018 - 29.67""")
    st.write("""California: 2015 - 17.01, 2016 - 24.77, 2017 - 26.79, 2018 - 31.43
    """)
    st.write("""Kết luận:
    Đánh giá theo số lượng bơ đã bán và doanh thu bơ 'Hass' ở các vùng khác nhau từ tháng 1/2015 đến 3/2018, 3 vùng California, West và Northeast có doanh thu cao và vượt xa các vùng còn lại. Vì vậy chiến lược phát triển trang trại trồng và tăng sản lượng bơ sẽ tập trung vào 3 vùng trên.""")
    st.write("""Thống kê theo từng loại bơ thì sản lượng và doanh thu bơ thường chiểm tỷ trọng và ảnh hưởng trực tiếp đến đường cong tăng trưởng của lĩnh vực kinh doanh bơ tại Hass. Vì vậy, tập trung phát triển và mổ rộng mô hình trang trại đối với bơ thường ở vùng trọng điểm nêu trên.""")

    st.write("""Bài phân tích cũng đánh giá riêng doanh thu của bơ hữu cơ theo từng vùng, vì giá trùng bình của bơ hưu cơ cao hơn nhiều so với bo thường nên việc lựa chọn vùng trọng điểm để phát triển mở rộng sản lượng bơ hữu cơ cũng được quan tâm.
    Từ kết quả thống kê trên, doanh thu bơ hữu cơ có xu hướng tăng cao ở một số vùng từ năm 2017. Trong đó, Northeast là một trong 10 vùng có tỷ lệ tăng trưởng doanh thu bơ hữu cao 12% so với năm 2016 và 2015. Vì vậy đề xuất phát triển mở rộng bơ hữu cơ ở Northeast.""")
    st.caption("""Tổng kết:
    Chiến lược mở rộng trang trại và sản xuất ở 3 vùng: California, West, Northeast.
    Phát triến bơ thường ở 3 vùng trên.
    Phát triển bơ hữu cơ ở Northeast.
    """)
    #size_bags
    size_bags = data[['SmallBags','LargeBags','type','year']]
    size_bags = size_bags.groupby(by = ['year','type']).mean().reset_index()
    size_conven = size_bags[size_bags['type']=='conventional']
    size_conven.index = size_conven['year']
    size_conven.drop(columns = ['type','year'], inplace = True)
    size_organic = size_bags[size_bags['type']=='organic']
    size_organic.index = size_organic['year']
    size_organic.drop(columns = ['type','year'], inplace = True)
    st.subheader('Total volumne by bags on organic avocado')
    st.dataframe(size_organic)
    st.subheader('Total volumne by bags on conventional avocado')
    st.dataframe(size_conven)
    st.write('#####Số lượng bơ đã bán với small bags chiểm tỷ trọng cao hơn nhiều so với large bags ở cả 2 loại bơ.')
    #PLU
    PLU = data[['4046','4225','4770','type','year']]
    PLU = PLU.groupby(by = ['year','type']).mean().reset_index()
    PLU_conven = PLU[PLU['type']=='conventional']
    PLU_conven.index = PLU_conven['year']
    PLU_conven.drop(columns = ['type','year'], inplace = True)
    PLU_organic = PLU[PLU['type']=='organic']
    PLU_organic.index = PLU_organic['year']
    PLU_organic.drop(columns = ['type','year'], inplace = True)
    st.subheader(' PLU of conventional avocado')
    st.table(PLU_conven)
    st.subheader(' PLU of organic avocado')
    st.table(PLU_organic)
    st.write('Số lượng bơ đã bán tập trung vào 2 PLU là 4046, 4225 của loại bơ thường.')
    st.write('Bơ thường số lượng đã bán của PLU 4046 và 4225 tương đương nhau.')
    st.write('Bơ hữu cơ số lượng đã bán của PLU 4225 chiếm tỷ trọng cao hơn so với 2 PLU còn lại.')
elif choice=='Business strategy':
    st.subheader('Summary')
    st.markdown('Áp dụng thống kê và phân tích giá trung bình, số lượng bơ đã bán và doanh thu tại các vùng khác nhau, kết quả ghi nhận doanh thu 2 loại bơ có xu hướng tăng trưởng ở các vùng California, West và Northeast. Trong đó, bơ loại thường có doanh số bán hàng cao vượt bậc so với bơ hữu cơ từ 1/2015 đến 3/2018. Mặt khác bơ thường có giá trùng thấp hơn so với bơ hữu cơ, tuy nhiên vì doanh số cao hơn nên ảnh hưởng trực tiếp đến doanh thu của bơ thường. Với kết quả thống kê và phân tích trên, áp dụng model RandomForest và Facebook Prophet cho việc dự đoán giá trung bình của 2 loại bơ tại các vùng Cailifornia, West, Northeast, cũ thể:')
    st.markdown('California và Northeast áp dụng dự đoán trên cả 2 loại bơ là thông thường và hữu cơ.')
    st.markdown('West áp dụng dự đoán trên bơ loại thông thường.')
    st.markdown('Bên cạnh đó, giá trung bình của 2 loại bơ đều có xu hương biến động theo mùa vụ. Giá trung bình tăng mạnh trong khoảng thời gian từ tháng 8 đến tháng 11 hàng năm, và bắt đầu giảm từ tháng 12 của năm trước đến tháng 2 của năm sau. Đặc biệt từ năm 2015 đến 2018 giá trung bình của 2 loại bơ tăng mạnh và đạt đỉnh vào tháng 9 hàng năm, và giảm sâu vào tháng 1 hàng năm.')
    st.markdown('Từ kết quả dự đoán, California và West đều có xu hướng giá trùng bình của bơ thường tăng dần đều qua các năm. Mặt khác, giá trung bình bơ thường tại vùng Northeast có xu hướng giảm mạnh trong vòng 5 năm tới.')
    st.markdown('Với kết quả dự đoán bơ hữu cơ, giá trung bình tại vùng California tặng chậm qua các năm. Tuy nhiên, giá trung bình tại vùng Northeast giảm mạnh qua các năm.')
    st.subheader('Chiến lược phát triển')
    st.write('Từ kết quả thống kê và phân tích, xác định chiến lược phát triển mở rộng trang trại và sản lượng bán bơ trong những năm tiếp theo của công ty Hass là tập trung phát triển và mở rộng bơ loại thường ở vùng California và West, bơ loại hữu cơ tại vùng Califonia. Mặt khác, trong thời gian 5 năm tới nên tập trung phát triên bơ thường tại các vùng.')
    st.write('Khuyến nghị thực hiện mở rộng dữ liệu chuyên về bơ hữu cơ tại các vùng khác nhau, để thực hiện thống kê và phân tích chuyên hơn về hữu cơ để xác định được vùng phát triển trọng tâm cho dòng bơ này. Vì giá trung bình hiện tại của bơ hữu cơ đang cao hơn nhiều so với bơ thường, tuy nhiên việc tăng sản lượng bơ hữu bằng 30% sản lượng bơ thường sẽ mang lại doanh thu lợi nhuận tốt hơn.')
    st.write('############################################')
