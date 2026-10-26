from langchain_community.tools import DuckDuckGoSearchRun
from app.tools.stock_price_tool import get_stock_prices
from app.tools.calculator_tool import expression_calculator_tool
from app.tools.rag_tool import rag_tool


search_tool=DuckDuckGoSearchRun()


tools=[get_stock_prices,search_tool,expression_calculator_tool,rag_tool]