import time
import pandas as pd
import streamlit as st
import scorecardpy as sc
import pickle
from streamlit_echarts import st_echarts


st.title('Scoreboard')

options_sex = ['Masculino', 'Femenino']

# Cargar el objeto Scorecard desde el archivo
with open('scorecard_points.pkl', 'rb') as file:
    scorecard = pickle.load(file)

with st.sidebar:
    
    limit_balance = st.number_input('Ingrese la cantidad de balance',min_value=0,step=1,key='limit_balance')
    
    sex = st.selectbox(
    'Genero',
       options_sex
    )
    
    age = st.number_input('Ingrese la edad', min_value= 18, step=1, key='age')

    pay_2 = st.number_input('mes pagado Agosto', min_value= -1, max_value=9, step=1, key='pay_2')
    pay_3 = st.number_input('mes pagado Julio', min_value= -1, max_value=9, step=1, key='pay_3')
    pay_4 = st.number_input('mes pagado Junio', min_value= -1, max_value=9, step=1, key='pay_4')
    pay_5 = st.number_input('mes pagado Mayo', min_value= -1, max_value=9, step=1, key='pay_5')
    pay_6 = st.number_input('mes pagado Abril', min_value= -1, max_value=9, step=1, key='pay_6')

    # BILL_AMT
    bill_amt1 = st.number_input('Amount of bill statement in September, 2005 (NT dollar)', min_value=0.0, step=1.0, key='bill_amt1')
    bill_amt2 = st.number_input('Amount of bill statement in August, 2005 (NT dollar)', min_value=0.0, step=1.0, key='bill_amt2')
    bill_amt3 = st.number_input('Amount of bill statement in July, 2005 (NT dollar)', min_value=0.0, step=1.0, key='bill_amt3')
    bill_amt4 = st.number_input('Amount of bill statement in June, 2005 (NT dollar)', min_value=0.0, step=1.0, key='bill_amt4')
    bill_amt5 = st.number_input('Amount of bill statement in May, 2005 (NT dollar)', min_value=0.0, step=1.0, key='bill_amt5')
    bill_amt6 = st.number_input('Amount of bill statement in April, 2005 (NT dollar)', min_value=0.0, step=1.0, key='bill_amt6')

    # PAY_AMT
    pay_amt1 = st.number_input('Amount of previous payment in September, 2005 (NT dollar)', min_value=0.0, step=1.0, key='pay_amt1')
    pay_amt2 = st.number_input('Amount of previous payment in August, 2005 (NT dollar)', min_value=0.0, step=1.0, key='pay_amt2')
    pay_amt3 = st.number_input('Amount of previous payment in July, 2005 (NT dollar)', min_value=0.0, step=1.0, key='pay_amt3')
    pay_amt4 = st.number_input('Amount of previous payment in June, 2005 (NT dollar)', min_value=0.0, step=1.0, key='pay_amt4')
    pay_amt5 = st.number_input('Amount of previous payment in May, 2005 (NT dollar)', min_value=0.0, step=1.0, key='pay_amt5')
    pay_amt6 = st.number_input('Amount of previous payment in April, 2005 (NT dollar)', min_value=0.0, step=1.0, key='pay_amt6')

if sex == options_sex[0]:
    value_sex = 1
elif sex == options_sex[1]:
    value_sex = 2


data = {
    'limit_bal': [limit_balance],
    'sex': [value_sex],
    'age': [age],
    'pay_2': [pay_2],
    'pay_3': [pay_3],
    'pay_4': [pay_4],
    'pay_5': [pay_5],
    'pay_6': [pay_6],
    'bill_amt1': [bill_amt1],
    'bill_amt2': [bill_amt2],
    'bill_amt3': [bill_amt3],
    'bill_amt4': [bill_amt4],
    'bill_amt5': [bill_amt5],
    'bill_amt6': [bill_amt6],
    'pay_amt1': [pay_amt1],
    'pay_amt2': [pay_amt2],
    'pay_amt3': [pay_amt3],
    'pay_amt4': [pay_amt4],
    'pay_amt5': [pay_amt5],
    'pay_amt6': [pay_amt6],
}

df_credit = pd.DataFrame(data)

puntaje_scores = sc.scorecard_ply(df_credit, scorecard, only_total_score=True)


def render_ring_gauge(value):
    option = {
        'series': [
            {
                'type': 'gauge',
                'center': ['50%', '50%'],
                'startAngle': 190,
                'endAngle': -10,
                'min': 0,
                'max': 1000,
                'axisLine': {
                    'lineStyle': {
                        'width': 20,
                        'color':[
                            [0.2, "#FF6666"],  
                            [0.4, "#FFB266"],  
                            [0.6, "#FFFF66"],  
                            [0.8, "#B2FF66"], 
                            [1, "#66FF66"],  
                        ],
                    },
                },
                'pointer': {
                    'icon': 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
                    'length': '12%',
                    'width': 20,
                    'offsetCenter': [0, '-60%'],
                    'itemStyle': {
                        'color': 'auto'
                    }
                },
                'axisTick': {
                    'show': False
                },
                'splitLine': {
                    'show': False
                },
                'axisLabel': {
                    'distance': 10,
                    'color': '#999',
                    'fontSize': 20
                },
                'detail': {
                    'valueAnimation': True,
                    'fontSize': 75,
                    'offsetCenter': [0, '5%']
                },
                'data': [
                    {
                        'value': value,
                    }
                ]
            }
        ]
    }

    st_echarts(option, height="700px", key="echarts")

# Run the ring gauge render function

render_ring_gauge(puntaje_scores.score.iloc[0])


# Add a delay and rerun the script to update values
time.sleep(2)

st.rerun()
