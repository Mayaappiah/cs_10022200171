@echo off
echo Starting ACity RAG Chatbot...
"%LOCALAPPDATA%\Programs\Python\Python313\python.exe" -m streamlit run app.py --server.port 8501
pause
