
# import streamlit as st
# import yfinance as yf
# import plotly.graph_objects as go
# import pandas as pd
# from datetime import datetime
# from pinecone import Pinecone
# from sentence_transformers import SentenceTransformer
# from groq import Groq
# from newsapi import NewsApiClient
# from dotenv import load_dotenv
# import os

# # Load environment variables from the .env file (if present)
# load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# st.set_page_config(layout="wide", page_title="Market Insider", page_icon="📈")

# def get_stock_info(symbol: str) -> dict:
#   """
#   Retrives and formats detailed information about a sotck from Yahoo Finance.

#   Args:
#     symbol (str): The stock ticker symbol to look up.

#   Returns:
#     dict: A dictionary containing detailed stock information, including ticker,
#           name, business summary, city, state, country, industry, and sector.
#   """

#   data = yf.Ticker(symbol)
#   info = data.info

#   return {
#       "name": info.get("shortName", "N/A"),
#       "summary": info.get("longBusinessSummary", "N/A"),
#       "sector": info.get("sector", "N/A"),
#       "industry": info.get("industry", "N/A"),
#       "market_cap": info.get("marketCap", "N/A"),
#       "price": info.get("currentPrice", "N/A"),
#       "revenue_growth": info.get("revenueGrowth", "N/A"),
#       "recommendation": info.get("recommendationKey", "N/A"),
#   }

# def format_large_number(num):
#     if num == "N/A":
#         return "N/A"
#     try:
#         num = float(num)
#         if num >= 1e12:
#             return f"${num/1e12:.1f}T"
#         elif num >= 1e9:
#             return f"${num/1e9:.1f}B"
#         elif num >= 1e6:
#             return f"${num/1e6:.1f}M"
#         else:
#             return f"${num:,.0f}"
#     except:
#         return "N/A"

# def format_percentage(value):
#     if value == "N/A":
#         return "N/A"
#     try:
#         return f"{float(value)*100:.1f}%"
#     except:
#         return "N/A"

# # "Buy", "Hold", "Sell", "Strong Buy", "Strong Sell", and sometimes "Underperform" or "Outperform"
# def format_recommendation(value):
#   match value:
#     case "NONE":
#       return "N/A"
#     case "STRONG_BUY":
#       return "BUY"
#     case "STRONG_SELL":
#       return "SELL"
#     case "UNDERPERFORM":
#       return "HOLD"
#     case "OUTPERFORM":
#       return "HOLD"
    
#   return value
    

# def stock_data_card(data, ticker):

#   with st.container():
#       st.markdown("""
            
#       """, unsafe_allow_html=True)

#       st.markdown(f"""
#           ### {data['name']} ({ticker})

#           **{data['sector']} | {data['industry']}**

#           {data['summary'][:60]}...
#       """, unsafe_allow_html=True)

#       metrics = [
#           {"label": "Market Cap", "value": format_large_number(data['market_cap'])},
#           {"label": "Price", "value": format_large_number(data['price'])},
#           {"label": "Growth", "value": format_percentage(data['revenue_growth'])},
#           {"label": "Rating", "value": format_recommendation(data['recommendation'].upper())}
#       ]

#       for metric in metrics:
#         st.metric(label=metric['label'], value=metric['value'], delta=None)

#       st.markdown("", unsafe_allow_html=True)

# st.title("Market Insider")
# st.write("Discover and compare stocks traded on the NYSE")

# user_query = st.text_input("Search for stocks by description, sector, or characteristics:")

# if st.button("🔍 Find Stocks", type="primary"):
#   with st.spinner("Searching..."):
#     client = Groq(
#       api_key=GROQ_API_KEY
#     )

#     system_prompt = f"""You are a prompt expert. Convert the user's stock search query into a more searchable format to be like more descriptive like a summary of a company's bussines. This query will be used to search for stocks using embeddings in a vector database and match it with the bussiness summaries of companies in the database. Keep the enhanced query concise. ONLY return the enhanced query nothing else, don't add anything else"""

#     llm_response = client.chat.completions.create(
#       model="llama-3.1-8b-instant",
#       messages=[{
#                   "role": "system", 
#                   "content": system_prompt
#                 },
#                 {
#                   "role": "user", 
#                   "content": f"Convert this stock search query into a searchable format to match the bussines summary of companise , ONLY return the query don't write anything except the query , just the query: {user_query}"
#                 }]
#     )

#     enhanced_query = llm_response.choices[0].message.content

#     # st.write(f"Enhanced Query: {enhanced_query}")

#     # Setup Pinecone
#     pc = Pinecone(api_key=PINECONE_API_KEY)
#     index_name = "ai-financial-analysis"
#     namespace = "stock-descriptions"
#     pinecone_index = pc.Index(index_name)

#     # Query Pinecone vectors
#     model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
#     query_embedding = model.encode(enhanced_query)
#     search_results = pinecone_index.query(
#       vector=query_embedding.tolist(),
#       top_k=5,
#       include_metadata=True,
#       namespace=namespace
#     )

#     ticker_list = [item['id'] for item in search_results['matches']]

#     stock_data = []
#     for ticker in ticker_list:
#       data = get_stock_info(ticker)
#       if data:
#         stock_data.append(data)
    
#     cols = st.columns(5)
#     for i in range(len(stock_data)):
#       with cols[i]:
#         stock_data_card(stock_data[i], ticker_list[i])

#     # Chart for comparison
#     if len(stock_data) > 0:
#       st.subheader("Stock Price Comparison")
#       fig = go.Figure()

#       for i, ticker in enumerate(ticker_list):
#         stock = yf.Ticker(ticker)
#         hist_data = stock.history(period="1y")

#         # Normalize the prices to percentage change
#         if not hist_data.empty:
#           hist_data['Normalized'] = (hist_data['Close'] / hist_data['Close'].iloc[0] - 1) * 100

#           fig.add_trace(go.Scatter(
#             x=hist_data.index,
#             y=hist_data['Normalized'],
#             name=f"{ticker}",
#             mode='lines'
#           ))

#       fig.update_layout(
#           title="1-Year Price Performance Comparison (%)",
#           yaxis_title="Price Change (%)",
#           template="plotly_dark",
#           height=500,
#           showlegend=True
#       )

#       st.plotly_chart(fig, use_container_width=True)

#       with st.spinner("Generating AI Analysis..."):
#         stock_info = "\n".join([
#               f"Stock: {data['name']} ({ticker_list[i]})\n"
#               f"Sector: {data['sector']}\n"
#               f"Price: {format_large_number(data['price'])}\n"
#               f"Market Cap: {format_large_number(data['market_cap'])}\n"
#               f"Growth: {format_percentage(data['revenue_growth'])}\n"
#               f"Recommendation: {data['recommendation']}\n"
#               f"Summary: {data['summary']}\n"
#               for i, data in enumerate(stock_data)
#             ])

#         chat_prompt = f"""Based on the user's query: "{user_query}"
#                           Here's the information about the matching stocks: {stock_info}

#                           Please provide a detailed analysis of these stocks, including:
#                           1. Why they match the user's query
#                           2. Key strengths and potential risks
#                           3. Comparative analysis between the stocks
#                           4. Investment considerations

#                           Format the response in a clear, organized way with sections and bullet points where appropriate.
#                         """

#         client = Groq(
#           api_key=GROQ_API_KEY
#         )
        
#         chat_response = client.chat.completions.create(
#                 model="llama-3.1-70b-versatile",
#                 messages=[
#                     {"role": "system", "content": "You are an expert stock analyst who always provides detailed, professional analysis."},
#                     {"role": "user", "content": chat_prompt}
#                 ])
        
#         analysis = chat_response.choices[0].message.content
#         st.subheader("Stock Analysis")
#         st.write(analysis)

#       st.subheader("Relevant Market News")

#       for ticker in ticker_list:
#         with st.spinner(f"Loading news for {ticker}..."):
#           news_items = newsapi.get_everything(
#               q=f"{ticker} stock",
#               language='en',
#               sort_by='relevancy',
#               page_size=2
#           )

#           if len(news_items["articles"]) > 0:
#             st.write(f"**Latest news for {ticker}**")
#           for article in news_items['articles']:
#             with st.expander(f"{article['title']}"):
#               st.write(article['description'])
#               st.write(f"**Source:** {article['source']['name']} | **Published on** {article['publishedAt']}")
#               st.link_button("Read full article", article['url'])


# --------------------------------------------------------------------------------
# import streamlit as st
# import yfinance as yf
# import plotly.graph_objects as go
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
# from groq import Groq
# from newsapi import NewsApiClient
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# st.set_page_config(layout="wide", page_title="Market Insider", page_icon="📈")

# def get_stock_info(symbol: str) -> dict:
#     """Fetch detailed stock information from Yahoo Finance."""
#     data = yf.Ticker(symbol)
#     info = data.info

#     return {
#         "name": info.get("shortName", "N/A"),
#         "summary": info.get("longBusinessSummary", "N/A"),
#         "sector": info.get("sector", "N/A"),
#         "industry": info.get("industry", "N/A"),
#         "market_cap": info.get("marketCap", "N/A"),
#         "price": info.get("currentPrice", "N/A"),
#         "revenue_growth": info.get("revenueGrowth", "N/A"),
#         "recommendation": info.get("recommendationKey", "N/A"),
#     }

# def search_tickers_by_query(query: str) -> list:
#     """
#     Dummy search function: fetch stock tickers based on the query.
#     Replace this with a mapping of queries to tickers or custom logic.
#     """
#     # Example hardcoded tickers for now
#     return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # Top stocks to show for testing

# # Helper functions for formatting
# def format_large_number(num):
#     if num == "N/A":
#         return "N/A"
#     try:
#         num = float(num)
#         if num >= 1e12:
#             return f"${num/1e12:.1f}T"
#         elif num >= 1e9:
#             return f"${num/1e9:.1f}B"
#         elif num >= 1e6:
#             return f"${num/1e6:.1f}M"
#         else:
#             return f"${num:,.0f}"
#     except:
#         return "N/A"

# def format_percentage(value):
#     if value == "N/A":
#         return "N/A"
#     try:
#         return f"{float(value)*100:.1f}%"
#     except:
#         return "N/A"

# st.title("Market Insider")
# st.write("Discover and compare stocks traded on the NYSE")

# # Input for search
# user_query = st.text_input("Search for stocks by description, sector, or characteristics:")

# if st.button("🔍 Find Stocks", type="primary"):
#     with st.spinner("Processing query..."):
#         # Use Groq to enhance the query
#         client = Groq(api_key=GROQ_API_KEY)
#         system_prompt = (
#             "You are a prompt expert. Convert the user's stock search query into a concise, descriptive format "
#             "that can be used to search for matching stocks. Only return the enhanced query."
#         )
#         llm_response = client.chat.completions.create(
#             model="llama-3.1-8b-instant",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_query}
#             ]
#         )
#         enhanced_query = llm_response.choices[0].message.content

#         # Fetch tickers for the query (replace with real logic if needed)
#         ticker_list = search_tickers_by_query(enhanced_query)

#     # Fetch stock data
#     stock_data = []
#     for ticker in ticker_list:
#         data = get_stock_info(ticker)
#         if data:
#             stock_data.append(data)

#     # Display results
#     cols = st.columns(len(stock_data))
#     for i, data in enumerate(stock_data):
#         with cols[i]:
#             st.subheader(f"{data['name']} ({ticker_list[i]})")
#             st.write(f"**Sector:** {data['sector']} | **Industry:** {data['industry']}")
#             st.write(f"**Summary:** {data['summary'][:100]}...")
#             st.metric("Price", format_large_number(data['price']))
#             st.metric("Market Cap", format_large_number(data['market_cap']))
#             st.metric("Growth", format_percentage(data['revenue_growth']))
#             st.metric("Recommendation", data['recommendation'])

#     # Price performance chart
#     if len(stock_data) > 0:
#         st.subheader("Stock Price Comparison")
#         fig = go.Figure()
#         for ticker in ticker_list:
#             stock = yf.Ticker(ticker)
#             hist_data = stock.history(period="1y")
#             if not hist_data.empty:
#                 hist_data['Normalized'] = (hist_data['Close'] / hist_data['Close'].iloc[0] - 1) * 100
#                 fig.add_trace(go.Scatter(
#                     x=hist_data.index,
#                     y=hist_data['Normalized'],
#                     name=ticker
#                 ))
#         fig.update_layout(
#             title="1-Year Price Performance (%)",
#             yaxis_title="Change (%)",
#             template="plotly_dark"
#         )
#         st.plotly_chart(fig)

#     # News section
#     st.subheader("Latest News")
#     for ticker in ticker_list:
#         news = newsapi.get_everything(q=f"{ticker} stock", language='en', sort_by='relevancy', page_size=2)
#         if news['articles']:
#             for article in news['articles']:
#                 st.write(f"**{article['title']}** - {article['source']['name']}")
#                 st.write(article['description'])
#                 st.write(f"[Read more]({article['url']})")


import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import json
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables from the .env file (if present)
load_dotenv()

st.set_page_config(layout="wide", page_title="Market Insider", page_icon="📈")

# Load the company tickers dataset
@st.cache_data
def load_company_tickers(filepath: str) -> dict:
    with open(filepath, "r") as file:
        return json.load(file)

company_tickers = load_company_tickers("company_tickers.json")

# Search companies by query
def search_companies(query: str, dataset: dict) -> list:
    """
    Search for companies based on a query.
    Args:
        query (str): The user-provided search term.
        dataset (dict): The loaded dataset of companies.

    Returns:
        list: A list of matching company tickers.
    """
    query = query.lower()
    results = []
    for _, company in dataset.items():
        if (
            query in company["title"].lower()
            or query in company["ticker"].lower()
        ):
            results.append(company["ticker"])
    return results[:5]  # Return top 5 matches

# Fetch stock information
def get_stock_info(symbol: str) -> dict:
    """
    Retrieves and formats detailed information about a stock from Yahoo Finance.
    Args:
        symbol (str): The stock ticker symbol to look up.
    Returns:
        dict: A dictionary containing detailed stock information.
    """
    data = yf.Ticker(symbol)
    info = data.info
    return {
        "name": info.get("shortName", "N/A"),
        "summary": info.get("longBusinessSummary", "N/A"),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "price": info.get("currentPrice", "N/A"),
        "revenue_growth": info.get("revenueGrowth", "N/A"),
        "recommendation": info.get("recommendationKey", "N/A"),
    }

# Helper functions for formatting
def format_large_number(num):
    if num == "N/A":
        return "N/A"
    try:
        num = float(num)
        if num >= 1e12:
            return f"${num/1e12:.1f}T"
        elif num >= 1e9:
            return f"${num/1e9:.1f}B"
        elif num >= 1e6:
            return f"${num/1e6:.1f}M"
        else:
            return f"${num:,.0f}"
    except:
        return "N/A"

def format_percentage(value):
    if value == "N/A":
        return "N/A"
    try:
        return f"{float(value)*100:.1f}%"
    except:
        return "N/A"

def format_recommendation(value):
    match value:
        case "NONE":
            return "N/A"
        case "STRONG_BUY":
            return "BUY"
        case "STRONG_SELL":
            return "SELL"
        case "UNDERPERFORM":
            return "HOLD"
        case "OUTPERFORM":
            return "HOLD"
    return value

# Display stock data card
def stock_data_card(data, ticker):
    with st.container():
        st.markdown(f"""
            ### {data['name']} ({ticker})
            **{data['sector']} | {data['industry']}**
            {data['summary'][:60]}...
        """, unsafe_allow_html=True)

        metrics = [
            {"label": "Market Cap", "value": format_large_number(data['market_cap'])},
            {"label": "Price", "value": format_large_number(data['price'])},
            {"label": "Growth", "value": format_percentage(data['revenue_growth'])},
            {"label": "Rating", "value": format_recommendation(data['recommendation'].upper())}
        ]

        for metric in metrics:
            st.metric(label=metric['label'], value=metric['value'], delta=None)

# Main app interface
st.title("Market Insider")
st.write("Discover and compare stocks traded on the NYSE")

user_query = st.text_input("Search for stocks by name, ticker, or characteristics:")

if st.button("🔍 Find Stocks"):
    with st.spinner("Searching for stocks..."):
        matching_tickers = search_companies(user_query, company_tickers)

        if not matching_tickers:
            st.warning("No matching stocks found. Please refine your search.")
        else:
            stock_data = []
            for ticker in matching_tickers:
                data = get_stock_info(ticker)
                if data:
                    stock_data.append(data)

            cols = st.columns(len(stock_data))
            for i in range(len(stock_data)):
                with cols[i]:
                    stock_data_card(stock_data[i], matching_tickers[i])

            # Chart for comparison
            if len(stock_data) > 0:
                st.subheader("Stock Price Comparison")
                fig = go.Figure()

                for i, ticker in enumerate(matching_tickers):
                    stock = yf.Ticker(ticker)
                   

