
import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.append('.')
sys.path.append('..')
from sentence_transformers import SentenceTransformer
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

openai_organization = os.getenv("openai_organization")
openai_api_key = os.getenv("openai_api_key")

pinecone_api_key = os.getenv("pinecone_api_key")
pinecone_env = os.getenv("pinecone_env")

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)


def find_match(input,top_k):
   
    model = SentenceTransformer('all-MiniLM-L6-v2')
    input_em = model.encode(input).tolist()
    index = pinecone.Index("open-source-v2")
    result = index.query(input_em, top_k=top_k, includeMetadata=True)
    matches = result['matches']
    print(matches)
    result_dict = {}
    result_dict['all_context'] = ''
    for i in range(min(top_k, len(matches))):
        meta = os.path.basename(matches[i]['metadata']['PDF Title'])
        context = matches[i]['metadata']['text']
        result_dict[f'meta_{i+1}'] = meta
        result_dict[f'url_{i+1}'] = matches[i]['metadata']['URL']
        # Publication Year': '2022'
        result_dict[f'publication_year_{i+1}'] = matches[i]['metadata']['Publication Year']
        # Reference (in APA 7 format)
        result_dict[f'reference_{i+1}'] = matches[i]['metadata']['Reference (in APA 7 format)']
        # country
        result_dict[f'country_{i+1}'] = matches[i]['metadata']['Country']
        # author
        result_dict[f'author_{i+1}'] = matches[i]['metadata']['Author']
        result_dict[f'context_{i+1}'] = context
        result_dict[f"chunk_id{i+1}"]=matches[i]['metadata']['chunk_id']
        
        result_dict['all_context'] += context + "\n"
    # print(matches[0])
    
    return result_dict
    # return matches




def extract_unique_chunks(input, top_k, multiplier):
    # Extract the relevant chunks from the results based on top_k * multiplier.
    result_dict = find_match(input, top_k * multiplier)
    
    # Initialize a dictionary to keep track of used PDFs.
    used = {}
    
    # Initialize a list to store relevant chunks.
    relevant_chunks = []
    
    # Initialize the unique_dict.
    unique_dict = {'all_context': ''}
    
    # Iterate over the results to find unique chunks.
    for i in range(0, top_k * multiplier + 1):
        pdf_name = result_dict.get(f'meta_{i}', None)  # Get the PDF name for the current chunk.
        
        # Check if the PDF name is not None and if it has not been used yet.
        if pdf_name is not None and pdf_name not in used:
            relevant_chunk = {
                'meta': pdf_name,
                
                'url': result_dict.get(f'url_{i}', None),  # Include the URL.
                'publication_year': result_dict.get(f'publication_year_{i}', None),  # Include the Publication Year.
                'reference': result_dict.get(f'reference_{i}', None),  # Include the Reference.
                'country': result_dict.get(f'country_{i}', None),  # Include the Country.
                'author': result_dict.get(f'author_{i}', None),  # Include the Author.
                'chunk_id': result_dict.get(f'chunk_id{i}', None),  # Include the chunk_id.
                'context': result_dict.get(f'context_{i}', None)  # Include the context.
                
            }
            relevant_chunks.append(relevant_chunk)
            used[pdf_name] = 1  # Mark the PDF as used.
            
            # Add the context to 'all_context'.
            unique_dict['all_context'] += relevant_chunk['context'] + "\n"
            
            # Add all relevant fields to unique_dict.
            for key, value in relevant_chunk.items():
                unique_dict[f'{key}_{len(relevant_chunks)}'] = value

            # If we have collected enough unique chunks, break the loop.
            if len(relevant_chunks) == top_k:
                break

    return unique_dict

