FROM python:3.11-slim

WORKDIR /app

COPY prod/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY prod/app.py .
COPY model/ model/

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
