import pathlib
import textwrap
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)
def gemini_response(context_12k,context_500k,query):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f""" Here are the "context" fetched from a company policies, "context" :{context_12k}
                                      and their is one more context fetched from the enviromnetal constraint a company
                                      needs to follow: {context_500k}. 
                                      Now i want you to answer the query: {query}, Make sure to answer in a friendly tone and be 
                                      informatibe and envronment friendly along with sticking to context provided.
                                      Always stick to the context provided to you and do not use your own knowledge base at all if the given query is not relevant to the query . 
                                      If you don't know the answer, just say that you don't know, don't try to make up an answer. 
                                      Use 6-7 sentences maximum. Keep the answer as concise as possible. 
                                      Always say 'thanks for asking!' at the end of the answer"""
                                    )
    return response.text

def test():
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Are you gemini pro or normal gemini?"
                                    )
    return response.text

print(test())
# if __name__ =="__main__":
#    main()
  