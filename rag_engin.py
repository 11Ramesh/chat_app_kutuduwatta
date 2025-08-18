# -*- coding: utf-8 -*-
import os
import asyncio
import difflib
import requests
import pandas as pd
import nest_asyncio
import re

from playwright.async_api import async_playwright

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_community.tools.openweathermap import OpenWeatherMapQueryRun
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS



# Step 1: Configure API Keys
os.environ["GOOGLE_API_KEY"] = "AIzaSyDjCU6IrOV6Xml84t7pJJy21v6QEPvnSb4"


# Step 2: Setup LLM and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
output_parser = StrOutputParser()


# Step 3: Load Data
csv_url = "https://docs.google.com/spreadsheets/d/1sUajk-3OCUJfXIwU0UqQ8uhW7PKxMXpi32O_KTASnD4/export?format=csv&gid=0"
df = pd.read_csv(csv_url)
text_google_sheet = df.to_string()
google_sheet_doc = Document(page_content=text_google_sheet, metadata={"source": "Google Sheet"})

pdf_url = "https://docs.google.com/document/d/18zz48wZ-ADyNcnGPmgbHVCkvAIochUjcJMrUlA-7yv0/export?format=txt"
Response = requests.get(pdf_url)
text_pdf = Response.text
pdf_doc = Document(page_content=text_pdf, metadata={"source": "PDF"})

docs = [google_sheet_doc, pdf_doc]


# Step 4: Split into chunks & create retriever
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever()


# Step 5: Prompt chain
template = """
Answer this question using the provided context only.

{question}

Context:
{context}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)


prompt_tem = ChatPromptTemplate.from_template("""
        You are a product name extractor.
        From the following question, return only the product name, without extra words.

        Question: "{question}"
        Output:
    """)
chain_tem = LLMChain(llm=llm, prompt=prompt_tem)



prompt_com = ChatPromptTemplate.from_template('''
                        You are an expert product price comparison assistant.
                        Your task is to compare two store data.

                        Question: "{question}"
                        Output:
                    ''')

chain_com = LLMChain(llm=llm, prompt=prompt_com)


# Step 6: Web scraping with Playwright
async def get_cargills_products(product_name_input):

    product_name_input_gen = chain_tem.run({"question": product_name_input})

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            await page.goto(
                f"https://cargillsonline.com/product/{product_name_input_gen}?PS={product_name_input_gen}",
                timeout=60000
            )

            await page.wait_for_selector("p.ng-binding[title]", timeout=5000)
            await page.wait_for_selector("h4.txtSmall.ng-binding", timeout=5000)

            quantity_selector = None
            if await page.query_selector("button.dropbtn.ng-binding.ng-scope"):
                quantity_selector = "button.dropbtn.ng-binding.ng-scope"
            elif await page.query_selector("button.dropbtn1.ng-binding.ng-scope"):
                quantity_selector = "button.dropbtn1.ng-binding.ng-scope"

            if quantity_selector:
                await page.wait_for_selector(quantity_selector)
                quantity_elements = await page.query_selector_all(quantity_selector)
            else:
                quantity_elements = []

            product_name_elements = await page.query_selector_all("p.ng-binding[title]")
            product_names = [await e.get_attribute("title") for e in product_name_elements if await e.get_attribute("title")]

            price_elements = await page.query_selector_all("h4.txtSmall.ng-binding")
            product_prices = [await e.inner_text() for e in price_elements if await e.inner_text()]

            product_quantities = [await e.inner_text() for e in quantity_elements if await e.inner_text()]

            count = min(len(product_names), len(product_prices), len(product_quantities))
            list_product = []

            if count == 0:
                print("No data found.")
            else:
                for i in range(count):
                    dic_product = {
                        "name": product_names[i].lower(),
                        "price": product_prices[i].lower(),
                        "quantity": product_quantities[i].lower()
                    }
                    list_product.append(dic_product)

                best_match = None
                best_score = 0
                for i in list_product:
                    score = difflib.SequenceMatcher(None, product_name_input_gen.lower(), i["name"]).ratio()
                    if score > best_score:
                        best_match = i
                        best_score = score

                if "compare" in product_name_input:
                    keyword = ['other','stores','another','compare','another','foodcity',]
                    text_s = " ".join(word for word in product_name_input.split() if word not in keyword)

                    our_store = chain.invoke(text_s)

                    
                    result = chain_com.run({
                        "question": f'{our_store} another shop {best_match["quantity"]} of {best_match["name"]} is {best_match["price"]}'
                    })
                    return result
                    
                else:
                     result = f'another shop {best_match["quantity"]} of {best_match["name"]} is {best_match["price"]}'
                     return result
        except Exception as e:
            result = "Input keyword is wrong:" 
            return result
          

        finally:
            await browser.close()
            
        


# Step 7: Smart Answer Function
def smart_answer(user_question):
    user_question = user_question.lower()
    if any(word in user_question for word in ["other stores", "foodcity", "another stores", "other shops", "compare","other shop", "another shop","others","anothershops","another","other"]):
        nest_asyncio.apply()
        return asyncio.run(get_cargills_products(user_question))
    else:
        return chain.invoke(user_question)


# Run in VS Code
if __name__ == "__main__":
    smart_answer("compare price of white rice in other shops")
