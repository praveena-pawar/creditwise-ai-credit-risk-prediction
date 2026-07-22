import streamlit as st

st.title("Test")

value = st.number_input(
    "Enter a number",
    min_value=0,
    max_value=100,
    value=10
)

st.write(value)