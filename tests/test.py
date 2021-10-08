from Broiler_Rates import head
import streamlit as st
def test_head():
    assert head() == st.header('Welcome To The "Predictive Analytics On Poultry".')