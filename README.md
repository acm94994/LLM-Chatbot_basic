# LLM-Chatbot_basic
1. install llama2 using ollama
2. pipenv install -r requirements.txt
3. pipenv run streamlit run final_chatbot.py

# How does this work?
A combination of a basic FAISS + BM25 keyword search. The retrievers are combined to make an ensemble retriever with a ratio that is changeable. The same is run on streamlit. 
Files need to be uploaded before asking questions.

# Possible improvements?
It takes too much time to work. Around 5-6 minutes for a single pdf + question. I can try finding some APIs and shortening the time significantly, like groqAPI.

