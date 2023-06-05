import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

import pickle as pkl
import streamlit as st

model = pkl.load(open("big_mart_model.pkl", "rb"))


st.title("  Bishop's Sales Predictions using ML")
# image
st.image("hero.jpg")

col1, col2, col3 = st.columns(3)

with col1:
    Item_Visibility = st.number_input(
        '**Item Visibility**', min_value=0.00, max_value=0.40, step=0.01)

with col1:
    Outlet_Location_Type_Numbers = st.selectbox(
        '**Outlet Location Type**', ['Tier 1', 'Tier 2', 'Tier 3'])

with col2:
    Item_MRP = st.number_input(
        '**Item MRP**', min_value=30.00, max_value=270.00, step=1.00)

with col2:
    Item_Fat_Content = st.selectbox(
        '**Item Fat Content**', ['Low Fat', 'Regular'])

with col3:
    Outlet_Size = st.selectbox('**Outlet Size**', ['Small', 'Medium', 'High'])

# with col3:


# Data Processing
data = {
    'Item_Visibility': Item_Visibility,
    'Item_MRP': Item_MRP,
    'Outlet_Size': Outlet_Size,
    'Outlet_Location_Type_Numbers': Outlet_Location_Type_Numbers,
    'Item_Fat_Content_Regular': Item_Fat_Content
}

oe = OrdinalEncoder(categories=[['Small', 'Medium', 'High']])
scaler = StandardScaler()


def make_prediction(data):
    df = pd.DataFrame(data, index=[0])

    if df['Item_Fat_Content_Regular'].values == 'Low Fat':
        df['Item_Fat_Content_Regular'] = 0.0

    if df['Item_Fat_Content_Regular'].values == 'Regular':
        df['Item_Fat_Content_Regular'] = 1.0

    df['Outlet_Location_Type_Numbers'] = df['Outlet_Location_Type_Numbers'].str.extract(
        '(\d+)', expand=False)
    df['Outlet_Size'] = oe.fit_transform(df[['Outlet_Size']])
    df[['Item_Visibility', 'Item_MRP']] = StandardScaler(
    ).fit_transform(df[['Item_Visibility', 'Item_MRP']])

    prediction = model.predict(df)

    return round(float(prediction), 2)


sales_prediction_output = ""
if st.button('**Predict Sales**'):
    sales_prediction = make_prediction(data)
    sales_prediction_output = f"**:blue[The sales is predicted to be {sales_prediction}]**"
    st.success(sales_prediction_output, icon="👍")
st.divider()
# Button for Prediction
# sales_prediction_output = ""


# if st.button('**Predict Sales**'):
#     sales_prediction = make_prediction(data)
#     sales_prediction_output = f"**:blue[The sales is predicted to be {sales_prediction}]**"
#     st.success(sales_prediction_output, icon="📊")
# st.divider()
