from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain

def demo_chatbot():
    demo_llm = ChatOpenAI(api_key = "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", model_name="gpt-3.5-turbo", temperature=0)
    return demo_llm

def demo_memory():
    llm_d = demo_chatbot()
    memory = ConversationBufferMemory(llm = llm_d, max_token_limit=500)
    return memory

def demo_conversation(input_text, memory):
    llm_d = demo_chatbot()

    prompt_template = PromptTemplate(
        input_variables=["user_input"],
        template="Reply to me in this exact format: 'Hi thanks for asking Raghavs AI!' {user_input}"
    )

    llm_conversation = LLMChain(
        prompt=prompt_template,
        llm=llm_d,
        memory=memory
    )
    return llm_conversation.run(user_input=input_text)
