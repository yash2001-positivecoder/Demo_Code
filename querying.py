from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

def get_500k_context(query,top_k):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    input_em = model.encode(query).tolist()
    pc = Pinecone(api_key="3f565c69-8c4d-4211-bca7-af58ac1b4e81")
    index = pc.Index("climate-gpt-yash-2")
    print("pinecone is setup")
    result = index.query(
    vector=input_em,
    # namespace="ns1",
    top_k=top_k,
    include_metadata = True
)
    data = result.matches 
    climate_data = ""
    for i in data:
        climate_data += i.metadata["text"]
    return result

def get_12k_context(query,top_k):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    input_em = model.encode(query).tolist()
    pc = Pinecone(api_key="5110c665-ceb5-46df-8f1d-ad8f07374b43")
    index = pc.Index("climate-gpt-yash")
    print("pinecone is setup")
    result = index.query(
    vector=input_em,
    # namespace="ns1",
    top_k=top_k,
    include_metadata = True
)
    data = result.matches 
    climate_data = ""
    for i in data:
        climate_data += i.metadata["text"]
    return result

if __name__ =="__main__":
    content_12k = get_12k_context("tata motors climate related issues",2)
    content_500k = get_500k_context("tata motors climate related issues",3)
    print(content_12k)
    print("*"*100)
    print(content_500k)

