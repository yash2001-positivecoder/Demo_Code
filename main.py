import streamlit as st
from querying import get_12k_context,get_500k_context
from gemini import gemini_response

def main():
    st.title("CorporateClimate GPT")

    query = st.text_input("Enter your query:")


    if st.button("Generate Response"):
        content_12k = get_12k_context(query, top_k=2)
        content_500k = get_500k_context(query,top_k=3)
        response = gemini_response(content_12k,content_500k, query)
        st.write("Generated Response:")
        st.write(response)

if __name__ == "__main__":
    main()
