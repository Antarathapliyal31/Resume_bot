from tavily import TavilyClient
from sentence_transformers import SentenceTransformer
import trafilatura
import requests
import regex as re
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import aisuite as ai
import os
from getpass import getpass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or getpass("Enter OpenAI API key: ")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or getpass("Enter Tavily API key: ")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = TavilyClient(api_key=TAVILY_API_KEY)
embedder=SentenceTransformer("all-MiniLM-L6-v2")
query=input("Enter your question")
query_vector=embedder.encode(query,normalize_embeddings=True).astype("float32").reshape(1, -1)
result=client.search(query,max_results=10)
#print(result)
links=[]
for item in result["results"]:
    links.append(item["url"])
def extract_content(url,timeout) -> str|None:
    headers={
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        # Request English pages—US first, any English next (q = preference)
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        response=requests.get(url,headers=headers,timeout=timeout)
        if response.status_code!=200:
            return None
        html=response.text
    except requests.RequestException:
        return None
    main_response=trafilatura.extract(html,include_comments=False,include_tables=False,include_links=True,include_formatting=True,output_format="markdown",favor_recall=True,deduplicate=True)
    return main_response

response=[]

for i in links:
    response_i=extract_content(i,20)
    if not response_i:
        continue
    response_i=re.sub(r"\n\n"," ",response_i)
    response_i=re.sub(r"\s+"," ",response_i).strip()
    response.append(response_i)
#print(response)
result_="\n".join(response)


chunks_obj=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=300,length_function=len, separators=["\n\n","\n",".","!","?",","," ",""])
chunks_=chunks_obj.split_text(result_)
#print(chunks_)
chunks_text_metadata=[]
for i, j in enumerate(chunks_):
    metadata = {"chunk_num": i}
    chunks_text_metadata.append({"text": j, "metadata": metadata})
#print(chunks_text_metadata)

chunk=[i["text"] for i in chunks_text_metadata]
vector=embedder.encode(chunk,normalize_embeddings=True)
vector=np.asarray(vector,dtype="float32")
dim=vector.shape[1]
index=faiss.IndexFlatIP(dim)
index.add(vector)
D,I=index.search(query_vector,k=3)
result_of_query=[chunks_text_metadata[i]["text"] for i in I[0]]
result_of_query=" ".join(result_of_query)

def prompt_(query,result_of_query):
    return (f""" Act as an assistant and write the resume solution for the query asked by the user.
    query={query}
    content={result_of_query}
    return answer as text""")


client_=ai.Client()
response__=client_.chat.completions.create(model="openai:gpt-4o-mini",messages=[{"role":"user","content":prompt_(query,result_of_query)}])
print(response__.choices[0].message.content)
