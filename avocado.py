from asyncio import futures
from click import option
import streamlit as st 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet 
from prophet.plot import add_changepoints_to_plot
from sklearn.metrics import r2_score
from math import sqrt
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric 
from prophet.plot import plot_plotly
import plotly.offline as py
import plotly.express as px
from PIL import Image
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.axes_grid1 import make_axes_locatable 

#1. Read Data
#Import data
data=pd.read_csv('avocado_new.csv')
image1 = Image.open('images.png')

#GUI
st.image(image1)
st.title("Data Science Project ")
st.header("Hass avocado price prediction")

#Upload file
upload_file = st.file_uploader('Choose a file', type = ['csv'])
if upload_file is not None:
    data = pd.read_csv(upload_file)
    data.to_csv('avocado_newdata.csv', index = False)
#-----------------------------------------------------------------------
#Visualization
q_averageprice = data.groupby(by = ['quarter'])['AveragePrice'].mean()



#2. Load Data
input = data[['TotalVolumne','4046','4225','4770','TotalBags','SmallBags','LargeBags','type']]
output = data[['AveragePrice']]

#3. Scale Data
scaler = RobustScaler()
df_scaler = scaler.fit_transform(input[['TotalVolumne','4046','4225','4770','TotalBags','SmallBags','LargeBags']])
df_scaler = pd.DataFrame(df_scaler,columns=['TotalVolumne','4046','4225','4770','TotalBags','SmallBags','LargeBags'])

encoder = LabelEncoder()
input['type_encoder'] = encoder.fit_transform(input['type'])
df_type = input['type_encoder']

#4. Buil Model - ExtraTreeRegressor
X = pd.concat([df_scaler,df_type],axis=1)
y = data['AveragePrice']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25,random_state=12)
ex = ExtraTreesRegressor()
model = ex.fit(X_train,y_train)

#5.Evaluate Model
yhat_train = model.predict(X_train)
yhat_test = model.predict(X_test)
R_square_train = round(model.score(X_train,y_train),3)
R_square_test = round(model.score(X_test,y_test),3)
RMSE = round(mean_squared_error(y_test,yhat_test,squared=False),3)

#-----------------------------------------------------------------------------------------
#6. Pre_data for facebookprophet

#5.Facebook Prophet Model
#Model
cali_cv = pd.read_csv('cali_cv_fb.csv')
cali_cv['ds'] = pd.to_datetime(cali_cv['ds'])
fbp_model = Prophet(interval_width=0.95,
                           weekly_seasonality=False,
                           daily_seasonality=False,
                           yearly_seasonality=True,
                           changepoint_prior_scale=0.001,
                           seasonality_prior_scale=0.1,
                           holidays_prior_scale=0.01,
                           changepoint_range=0.8,
                           growth='linear',
                           seasonality_mode='multiplicative'
                           )

cali_con_model = fbp_model.fit(cali_cv)

cali_or= pd.read_csv('cali_or_fb.csv')
cali_or['ds'] = pd.to_datetime(cali_or['ds'])
fbp_model1 = Prophet(interval_width=0.95,
                           weekly_seasonality=False,
                           daily_seasonality=False,
                           yearly_seasonality=True,
                           changepoint_prior_scale=0.001,
                           seasonality_prior_scale=0.1,
                           holidays_prior_scale=0.01,
                           changepoint_range=0.8,
                           growth='linear',
                           seasonality_mode='multiplicative'
                           )

cali_or_model = fbp_model1.fit(cali_or)

west_cv = pd.read_csv('west_cv_fb.csv')
west_cv['ds'] = pd.to_datetime(west_cv['ds'])
fbp_model2 = Prophet(interval_width=0.95,
                           weekly_seasonality=False,
                           daily_seasonality=False,
                           yearly_seasonality=True,
                           changepoint_prior_scale=0.001,
                           seasonality_prior_scale=0.1,
                           holidays_prior_scale=0.01,
                           changepoint_range=0.8,
                           growth='linear',
                           seasonality_mode='multiplicative'
                           )

west_model = fbp_model2.fit(west_cv)

sc_cv = pd.read_csv('sc_cv_fb.csv')
sc_cv['ds'] = pd.to_datetime(sc_cv['ds'])
fbp_model3 = Prophet(interval_width=0.95,
                           weekly_seasonality=False,
                           daily_seasonality=False,
                           yearly_seasonality=True,
                           changepoint_prior_scale=0.001,
                           seasonality_prior_scale=0.1,
                           holidays_prior_scale=0.01,
                           changepoint_range=0.8,
                           growth='linear',
                           seasonality_mode='multiplicative'
                           )

sc_model = fbp_model3.fit(sc_cv)

north_or = pd.read_csv('north_or_fb.csv')
north_or['ds'] = pd.to_datetime(north_or['ds'])
fbp_model4 = Prophet(interval_width=0.95,
                           weekly_seasonality=False,
                           daily_seasonality=False,
                           yearly_seasonality=True,
                           changepoint_prior_scale=0.001,
                           seasonality_prior_scale=0.1,
                           holidays_prior_scale=0.01,
                           changepoint_range=0.8,
                           growth='linear',
                           seasonality_mode='multiplicative'
                           )
north_model = fbp_model4.fit(north_or)




#---------------------------------------------------------------------------------
#Gui
st.sidebar.header('Project Understand')
menu = ['Business Objective','Statistic & Analysis','Build Model','End-user Application']
choice = st.sidebar.selectbox('Content', menu)
if choice == 'Business Objective':
    st.header('Business Objective')
    st.subheader('Business Understand')
    st.write('B?? Hass, m???t c??ng ty c?? tr??? s??? t???i Mexico, chuy??n s???n xu???t nhi???u lo???i qu??? b?? ???????c b??n ??? r???t nhi???u v??ng c???a n?????c M??? v???i 2 lo???i b?? l?? b?? th?????ng v?? b?? h???u c??. S???n xu???t v???i quy tr??nh ????ng g??i theo 3 ti??u chu???n: Small, Large v?? XLarge d???ng t??i, v???i 3 PLU (Product Look Up) l??: 4046,4225 v?? 4770.')
    st.write('Hi???n t???i, c??ng ty ch??a c?? m?? h??nh d??? ??o??n gi?? b?? cho vi???c m??? r???ng c??c lo???i trang tr???i B?? ??ang c?? cho vi???c tr???ng b?? ??? c??c v??ng kh??c')
    st.write('Goal: ')
    st.write(' - X??y d???ng m?? h??nh d??? ??o??n gi?? b?? Hass trung b??nh ??? M???')
    st.write(' - L???a ch???n v??ng v?? xem x??t vi???c m??? r???ng s???n xu???t, kinh doanh.')
    st.write('Scope: ')
    st.write("D??? li???u ???????c ghi nh???n tr???c ti???p t??? m??y t??nh ti???n c???a c??c nh?? b??n l??? v?? m??y qu??t b??n l??? h??ng tu???n cho l?????ng b??n l??? v?? gi?? b?? t??? th??ng 4/2015 ?????n 3/2018")
    st.subheader('Data Understand')
    st.write('Gi?? Trung b??nh (Average Price) trong b???ng ph???n ??nh gi?? tr??n m???t ????n v??? (m???i qu??? b??), ngay c??? khi nhi???u ????n v??? (b??) ???????c b??n trong bao')
    st.write('M?? tra c???u s???n ph???m - Product Lookup codes -PLUs- trong b???ng ch??? d??nh cho b?? Hass, kh??ng d??nh cho c??c s???n ph???m kh??c.')
    st.write('To??n b??? d??? li???u ???????c ????? ra v?? l??u tr??? trong t???p tin avocado.csv v???i 18249 record.')
    st.subheader('Acquire')
    st.write('B??i to??n 1: USA s Avocado AveragePrice Prediction S??? d???ng c??c thu???t to??n Regression nh?? Linear Regression, Random Forest, XGB Regressor...')
    st.write('B??i to??n 2: Conventional/Organic Avocado Average Price Prediction for the future in California/NewYork... - s??? d???ng c??c thu???t to??n Time Series nh?? ARIMA, Prophet..')
if choice =='Statistic & Analysis':
    st.header('Statistic & Analysis')
#Show data
    st.subheader("1.Data")
    if st.checkbox('Preview Dataset'):
        st.text('First 5 lines of dataset')
        st.dataframe(data.head())
        st.text('Last 5 lines of dataset')
        st.dataframe(data.tail())
    if st.checkbox('Info of dataset'):
        st.text('Describle of dataset')
        st.code(data.describe())
        
#Visualization Data
#Avg Price
    st.subheader('2.Data Visualization')
    st.write('2.1. Average Price')
    st.caption('Average Price by Quarter in Year from 2015 to 2018')
        
    q_avg = q_averageprice.plot(grid = True, figsize = (14,7), title = 'Average Price by Quarter in Year')
    st.pyplot(q_avg.figure)
     
    type_averageprice = data.groupby(by = ['type','quarter'])['AveragePrice'].mean().reset_index()
    type_averageprice['quarter'] = type_averageprice['quarter'].astype(str)
    f = plt.figure(figsize = (14,7))
    sns.lineplot(x = type_averageprice['quarter'], y = type_averageprice['AveragePrice'], hue = type_averageprice['type']).set_title('Average Price by quater & type')
    plt.show()
    st.pyplot(f.figure)
        
    st.write('2.2. Total Volumne')
    
    type_volumne = data.groupby(by = ['type','quarter'])['TotalVolumne'].sum().reset_index()
    type_volumne['quarter'] = type_volumne['quarter'].astype(str)
    f1 = plt.figure(figsize = (15,5))
    sns.barplot(x = type_volumne['quarter'], y = type_volumne['TotalVolumne'], hue = type_volumne['type'])
    plt.title('Total Volumne by type & quater in year')
    plt.show()
    st.pyplot(f1.figure)
    
    region_ = data.groupby(by = ['region','year'])['TotalVolumne'].sum().reset_index()
    region_.sort_values(by ='TotalVolumne', ascending = False, inplace = True)
    TotalUS = region_[region_['region']=='TotalUS']
    f2 = plt.figure(figsize = (15,5))
    sns.barplot(x = TotalUS['year'], y = TotalUS['TotalVolumne'], color = 'b').set_title('Total Volumne by Year in US (2015 - 2018)')
    plt.show()
    st.pyplot(f2.figure)
    
    state_big = region_.set_index('region')
    state_big = state_big.loc[['West','California','SouthCentral', 'Northeast','Southeast','GreatLakes', 'Midsouth', 'LosAngeles', 'Plains']]
    state_big.reset_index(inplace = True)
    f3 = plt.figure(figsize = (15,5))
    sns.barplot(x = state_big['region'], y = state_big['TotalVolumne'], hue = state_big['year']).set_title('Total Volumne by Year in State (2015 - 2018)')
    plt.show()
    st.pyplot(f3.figure)
    
    st.write('2.3. Sales')
    data['sales'] = data['TotalVolumne']*data['AveragePrice']
    region_sales = data.groupby(by = ['region','year','type'])['sales'].sum().reset_index()
    region_sales.sort_values(by = 'sales', ascending = False, inplace = True)
    region_sales.year = pd.to_datetime(region_sales.year, format='%Y')
    region_sales['year'] = pd.to_datetime(region_sales['year']).dt.year
    state_big_sales = region_sales.set_index('region')
    state_big_sales = state_big_sales.loc[['West','California','SouthCentral', 'Northeast','Southeast','GreatLakes', 'Midsouth', 'LosAngeles', 'Plains']]
    state_big_sales.reset_index(inplace = True)
    sales_con = state_big_sales[state_big_sales['type'] == 'conventional']
    sales_org = state_big_sales[state_big_sales['type'] == 'organic']
    
    f4 = plt.figure(figsize=(15,6))
    sns.lineplot(x = sales_con['year'], y = sales_con['sales'], hue = sales_con['region']).set_title('Sales of Conventional by year & region')
    plt.show()
    st.pyplot(f4.figure)
    
    f5 = plt.figure(figsize=(15,6))
    sns.lineplot(x = sales_org['year'], y = sales_org['sales'], hue = sales_org['region']).set_title('Sales of Organic by year & region')
    plt.show()
    st.pyplot(f5.figure)
          
#Build Model
if choice =='Build Model':
    st.subheader('3. Build Model')

    #Peformance of Model
    st.write('3.1. ExtraTreeRegressor - Average Price Prediction')
    st.write('Evalute model performance')
    st.write('R_square_train:',R_square_train )
    st.write('R_square_test:',R_square_test)
    st.write('RMSE:',RMSE)
    
    f7 = plt.figure(figsize=(10,5))
    plt.scatter(yhat_test, y_test)
    plt.xlabel('Model predictions')
    plt.ylabel('True values')
    plt.plot([0, 3], [0, 3], 'r-')
    plt.title('Evaluate Actual Value & Predict Value')
    plt.show()
    st.pyplot(f7.figure)
    
    f8 = plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    sns.kdeplot(y_train, color='blue', label='True train values')
    sns.kdeplot(yhat_train, color='red', label='Predict train values')
    plt.title('y_train vs yhat_train')
    plt.subplot(1,2,2)
    sns.kdeplot(y_test, color='blue', label='True test values')
    sns.kdeplot(yhat_test, color='red', label='Predict test values')
    plt.title('y_test vs yhat_test')
    plt.legend()
    plt.show()
    st.pyplot(f8.figure)

    st.write('3.2. Facebook Prophet - Strategic Region')
    select_region = st.selectbox('Strategic Development Region',['California','West','SouthCentral','NorthEast'])
    if select_region == 'California':
        st.write('MPEA of Conventional')
        per1 = Image.open('percali_con.png')
        st.image(per1)
        st.write('MAPE Organic')
        per2 = Image.open('percali_or.png')
        st.image(per2)
    if select_region == 'West':
        per3 =Image.open('west.png')
        st.image(per3)  
    if select_region =='SouthCentral':
        per4 = Image.open('sc.png')
        st.image(per4)
    if select_region =='NorthEast':
        per5 = Image.open('north.png')
        st.image(per5)
        
#End User Addition
if choice == 'End-user Application':      
    st.header('End-user Application')
    model_pre = ['Avg Price Prediction','Time Series FacebookProphet']
    choice_model = st.sidebar.selectbox('Model Prediction', model_pre)
    if choice_model == 'Avg Price Prediction':
        st.write('End-user Input Feature')
        #Collects user input feature into data
        uploaded_file = st.sidebar.file_uploader('Upload your input CSV file', type=['csv'])
        #Choose Upload or Input data
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
        else:
            def user_input_data():
                Total_Volumne = st.number_input('Total Volumne', min_value=0, value=0)
                PLU_4046 = st.number_input('PLU 4046', min_value=0, value=0)
                PLU_4225 = st.number_input('PLU 4225', min_value=0, value=0)
                PLU_4770 = st.number_input('PLU 4770', min_value=0, value=0)
                Total_Bags = st.number_input('Total Bags', min_value=0, value=0)
                Small_Bags = st.number_input('Small Bags', min_value=0, value=0)
                Large_Bags = st.number_input('Large Bags',min_value=0, value=0)
                Types = st.selectbox('Types',('conventional', 'organic'))
                data_new = {'TotalVolumne': Total_Volumne,
                            '4046': PLU_4046,
                            '4225': PLU_4225,
                            '4770':PLU_4770,
                            'TotalBags':Total_Bags,
                            'SmallBags':Small_Bags,
                            'LargerBags':Large_Bags,
                            'types':Types}
                data_input = pd.DataFrame(data_new, index=[0])
                columns = ['TotalVolumne','4046','4225','4770','TotalBags','SmallBags','LargerBags','types']
                data_input = data_input.reindex(columns = columns)
                return data_input
            with st.sidebar.form('my_form'):
                input_df = user_input_data()
                sub = st.form_submit_button('Submit')
        #Scale and encoder data       
        scale = scaler.transform(input_df[['TotalVolumne','4046','4225','4770','TotalBags','SmallBags','LargerBags']])
        scale = pd.DataFrame(scale,columns=['TotalVolumne','4046','4225','4770','TotalBags','SmallBags','LargerBags'])
        input_df['type_encoder'] = encoder.transform(input_df['types'])
        lb_encoder = input_df['type_encoder']
        X_new = pd.concat([scale,lb_encoder],axis=1)
        #Show prediction
        st.dataframe(input_df)  
        prediction = model.predict(X_new)
        st.write('Prediction:', prediction)
    elif choice_model == 'Time Series FacebookProphet':
        #Select input or upload data
        st.write('End-user Input Feature')
        ts = st.sidebar.number_input('Time prediction', min_value=0, value=0, max_value=10)
        
        #Select model 
        region_model = ['California Convetional Prediction', 'California Organic Prediction','West Conventional Prediction','SouthCentral Conventional Prediction','Northeast Organic Prediction']
        choice_region = st.sidebar.selectbox('Region', region_model)
        
        if choice_region =='California Convetional Prediction':
            ds = cali_con_model.make_future_dataframe(periods=48*ts, freq='W')
            forecast = cali_con_model.predict(ds)
            new_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']]
            
            fig = cali_con_model.plot(forecast)
            a = add_changepoints_to_plot(fig.gca(),cali_con_model,forecast)
            st.pyplot(fig.figure)
            st.dataframe(new_forecast)
            
        elif choice_region =='California Organic Prediction':
            ds = cali_or_model.make_future_dataframe(periods=48*ts, freq='W')
            forecast = cali_or_model.predict(ds)
            new_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']]
            
            fig = cali_or_model.plot(forecast)
            a = add_changepoints_to_plot(fig.gca(),cali_or_model,forecast)
            st.pyplot(fig.figure)
            st.dataframe(new_forecast)
            
        elif choice_region == 'West Conventional Prediction':
            ds = west_model.make_future_dataframe(periods=48*ts, freq='W')
            forecast = west_model.predict(ds)
            new_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']]
            
            fig = west_model.plot(forecast)
            a = add_changepoints_to_plot(fig.gca(),west_model,forecast)
            st.pyplot(fig.figure)
            st.dataframe(new_forecast)
            
        elif choice_region =='SouthCentral Conventional Prediction':
            ds = sc_model.make_future_dataframe(periods=48*ts, freq='W')
            forecast = sc_model.predict(ds)
            new_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']]
            
            fig = sc_model.plot(forecast)
            a = add_changepoints_to_plot(fig.gca(),sc_model,forecast)
            st.pyplot(fig.figure)
            st.dataframe(new_forecast)
            
        elif choice_region == 'Northeast Organic Prediction':
            ds = north_model.make_future_dataframe(periods=48*ts, freq='W')
            forecast = north_model.predict(ds)
            new_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']]
            fig = north_model.plot(forecast)
            a = add_changepoints_to_plot(fig.gca(),north_model,forecast)
            st.pyplot(fig.figure)
            st.dataframe(new_forecast)

                
