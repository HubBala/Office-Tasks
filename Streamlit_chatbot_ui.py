# my_chatbot UI by using streamlit. 

import streamlit as st
import requests

st.title("Rasa My_Chatbot UI")

user_input = st.text_input("you: ", "")

if st.button("Send"):
    if user_input:
        response = requests.post( 
             "http://localhost:5005/webhooks/rest/webhook",
            json={"sender": "user", "message": user_input}
        )

        # Parse and show bot response
        if response.status_code == 200:
            bot_responses = response.json()
            for bot_msg in bot_responses:
                st.text(f"Bot: {bot_msg.get('text')}")
        else:
            st.error("Error: Could not reach Rasa server.")