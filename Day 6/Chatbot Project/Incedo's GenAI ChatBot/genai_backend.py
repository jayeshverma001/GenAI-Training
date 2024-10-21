from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.prompts.few_shot import FewShotPromptTemplate

def demo_chatbot():
    demo_llm = ChatOpenAI(api_key = "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", model_name="gpt-3.5-turbo", temperature=0.5, top_p=0.5)
    return demo_llm

def demo_memory():
    llm_d = demo_chatbot()
    memory = ConversationBufferMemory(llm = llm_d, max_token_limit=500)
    return memory

def demo_conversation(input_text, memory):
    llm_d = demo_chatbot()
    
    example_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        
        #template='''Respond with: Sorry, the question is out of my scope. 
        #           if the question is not related to AI, Gen AI or bigdata. 
        #           If the question is anyway related to big data, AI, Gen AI: 
        #           Reply to me in this exact format: 'Hi thanks for asking Incedo's Gen AI! ' {question}\n{answer}"

        template="Reply to me in this exact format: 'Hi thanks for asking Incedo's Gen AI! ' {question}\n{answer}"
    )

    examples = [    
        {"question": "Who are you?", "answer": "I am Incedo's Gen AI ChatBot here to assist you with any queries related to Gen AI."},
        {"question": "What technologies Incedo use for developing Gen AI applications?", "answer": "We use 1. LangChain, 2. Streamlit and other libraries in python to build Gen AI applications for our clients."},
        {"question": "What is langchain?", "answer": "LangChain is a framework to build with LLMs by chaining interoperable components. Incedo uses this library extensively to build robust AI applications"},
    ]

    prompt_template = FewShotPromptTemplate(
        input_variables=["question"],
        suffix="Reply to me in this exact format: Hi thanks for asking Incedo's Gen AI! {question}\n",
        example_prompt=example_prompt,
        examples=examples
    )

    llm_conversation = LLMChain(
        prompt=prompt_template,
        llm=llm_d,
        memory=memory
    )
    return llm_conversation.run(question=input_text)
