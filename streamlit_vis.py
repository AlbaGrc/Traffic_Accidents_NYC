import streamlit as st
from dashboard import *

st.set_page_config(layout="wide")

st.title("New York City Traffic Accidents")
st.subheader("Analysis of vehicle collisions through NYC open data")

st.markdown("---")

with st.sidebar:
    st.sidebar.title("About")
    st.sidebar.info(
        """
        Dashboard created by student Alba García Ochoa, for the Visual Analytics course in Sapienza University, Rome.
        
        **For contacting the author:**
        garciaochoa.2181770@studenti.uniroma1.it
        
        Source code: [GitHub](https://github.com/AlbaGrc/Traffic_Accidents_NYC)
        """
    )

st.altair_chart(final_dashboard)