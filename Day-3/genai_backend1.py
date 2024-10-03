import os
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

def demo_chatbot():
    demo_llm = Bedrock(
        credentials_profile_name= 'default',
        model_id= "meta.llama3-70b-instruct-v1:0",
        model_kwargs= {
            "temperature": 0.4,
            "top_p": 0.5,
            "max_gen_len": 500})
    return demo_llm
def demo_memory():
    llm_d = demo_chatbot()
    memory = ConversationBufferMemory(llm = llm_d, max_token_limit=500)
    return memory

def demo_conversation(input_text, memory):
    llm_chain_data = demo_chatbot()
    llm_conversation = ConversationChain(llm=llm_chain_data, memory=memory, verbose= True)
    chat_reply = llm_conversation.predict(input = input_text)
    return chat_reply
