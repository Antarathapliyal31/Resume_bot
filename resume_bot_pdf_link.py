from PyPDF2 import PdfReader
import regex as re
from bs4 import BeautifulSoup
import requests
from faiss import IndexFlatIP
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import aisuite as ai
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or getpass("Enter OpenAI API key: ")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
reader=PdfReader("/Users/antarathapliyal/Desktop/resume_chatbot_docs/harvard_resume_tips.pdf")
raw_text=""
for i in reader.pages:
    raw_text+=i.extract_text()

clean_text=re.sub(r"RESUMES AND COER LETTERS\s*","",raw_text)
clean_text = re.sub(r"(https?:\/\/)?\S+\.\S+", '', clean_text)
clean_text=re.sub(r"\s+"," ",clean_text)
print(clean_text[:500])

raw_text1=""
reader1=PdfReader("/Users/antarathapliyal/Desktop/resume_chatbot_docs/resume_action_verbs.pdf")
for i in reader1.pages:
    raw_text1+=i.extract_text()

clean_text1=re.sub(r"\s+"," ",raw_text1)


raw_text2=""
reader2=PdfReader("/Users/antarathapliyal/Desktop/resume_chatbot_docs/resume_keywords_by_role.pdf")
for i in reader2.pages:
    raw_text2+=i.extract_text()

clean_text2=re.sub(r"\s+"," ",raw_text2)


raw_text3=""
reader3=PdfReader("/Users/antarathapliyal/Desktop/resume_chatbot_docs/resume_tips_custom.pdf")
for i in reader3.pages:
    raw_text3+=i.extract_text()

clean_text3=re.sub(r"\s+"," ",raw_text3)


html_code=requests.get("https://www.themuse.com/advice/43-resume-tips-that-will-help-you-get-hired").text
structured_html=BeautifulSoup(html_code,"html.parser")
#print(structured_html.head)
text_p_lst=[i.text for i in structured_html.find_all("p")]
text_p=" ".join(text_p_lst)
print(text_p)

html_code2=requests.get("https://www.linkedin.com/business/learning/blog/career-success-tips/how-to-write-a-resume-that-will-actually-get-a-recruiter-s-atten").text
structured_html2=BeautifulSoup(html_code2,"html.parser")
#print(structured_html.head)
text_p_lst2=[i.text for i in structured_html2.find_all("p")]
text_p2=" ".join(text_p_lst2)
text_p2=re.sub(r"Career success tips","",text_p2).strip()
text_p2=re.sub(r"\s+"," ",text_p2)
print(text_p2)


chunks_text_metadata=[]
chunks_obj=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=300,length_function=len, separators=["\n\n","\n",".","!","?",","," ",""])
clean_txt_dict={"harvard_resume_tips":clean_text,"resume_action_verbs":clean_text1,"resume_keywords_by_role":clean_text2,"resume_tips_custom":clean_text3,"resume_tips_weblink":text_p,"recruiter_attention":text_p2}
for index,value in clean_txt_dict.items():
    chunk_i=chunks_obj.split_text(value)
    for i,j in enumerate(chunk_i):
        metadata={"chunk_num":i,"source":value}
        chunks_text_metadata.append({"text":j,"metadata":metadata})


#process to get embeddings- 1. load the model 2.get the text 3.use the encode to get vectors 4. convert that to array with data type float32
embedder=SentenceTransformer("all-MiniLM-L6-v2")
chunk=[i["text"] for i in chunks_text_metadata]
print("chunk is",chunk)
vector=embedder.encode(chunk,normalize_embeddings=True)
#we are converting the output of embedder.encode to array as vector index database like FAISS requires embeddings in the form of array
vector=np.array(vector, dtype="float32")
print(vector)

#FAISS- Facebook AI similarity search, used for similarity search, it operates by storing vectors in DS called indexes which uses dot product to find similarity

dimension=vector.shape[1]
#It creates an index (a searchable collection of vectors),and it teels the way it compares vectors inside this index is using Inner Product (dot product)
index=faiss.IndexFlatIP(dimension)
index.add(vector)

question="How should i write my resume as an masters in computer science student"
resume_query=embedder.encode("How should i write my resume as an masters in computer science student",normalize_embeddings=True).astype("float32").reshape(1, -1)
# print(resume_query.shape)

D,I=index.search(resume_query,k=3)
#FAISS always returns results in 2-D shape:D shape = (num_queries, k), I shape = (num_queries, k),Even if you searched only one query, FAISS still wraps it in a list.
"""I =
[
   [ 23, 8, 17 ]     ← top-3 result indices for query #1
]"""
text_=[chunks_text_metadata[i]["text"] for i in I[0]]
# print(text_)
text_="\n\n".join(text_)
# print(text_)

def prompt_(query,content):
    return (f"""Act as a resume expert and provide your answer for the given query from the content you find.
     If you donot get a useful answer from the content reply i dont know. 
     content={content},query={query} Return answer as a text""")


client=ai.Client()
response=client.chat.completions.create(model="openai:gpt-4o-mini",messages=[{"role":"user","content":prompt_(question,text_)}])
print(response.choices[0].message.content)
