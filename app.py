import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform


plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

#title
st.title('Suv hayvonlarini klassifikatsiya qiluvchi model')

#rasmni yuklash
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)

    img = PILImage.create(file)

    model = load_learner('fish_model.pkl')

    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    fig=px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
    
