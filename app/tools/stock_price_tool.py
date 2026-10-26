from langchain_core.tools import tool
import requests 
from dotenv import load_dotenv
load_dotenv()
import os

api_key=os.getenv("stock_api")

@tool
def get_stock_prices(symbol:str):
    """
    fetch the latest stock price for a given symbol (e.g. 'AAPL','TSLA')
    using the Alpha Vantage with API key in the URL.
    """
    url=f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey={api_key}"
    r=requests.get(url)
    return r.json()
