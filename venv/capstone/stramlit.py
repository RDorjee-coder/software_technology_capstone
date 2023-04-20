import streamlit as st

st.title('My First Streamlit App')
button_clicked = st.button('Click me!')

if button_clicked:
    st.write('Hello, World!')