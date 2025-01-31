    import os
    from dotenv import load_dotenv, find_dotenv
    from langchain_community.tools.tavily_search import TavilySearchResults
    from typing_extensions import TypedDict
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.document_loaders import WebBaseLoader
    import operator
    from typing import Annotated, List
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_community.document_loaders import ArxivLoader
    from langchain_core.documents import Document
    from langchain_community.vectorstores import Chroma
    from langgraph.graph import StateGraph, START, END
    import streamlit as st
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    load_dotenv(find_dotenv())

    API_KEYS = ['LANGCHAIN_API_KEY', 'OPENAI_API_KEY', 'LANGCHAIN_TRACING_V2', 'LANGCHAIN_ENDPOINT', 
                'LANGCHAIN_PROJECT', 'TAVILY_API_KEY'
                ]

    for api_key in API_KEYS:
        os.environ[api_key] = os.getenv(api_key)

    st.title("🔍 Multi-Agent Research Assistant")
    user_question = st.text_input("Enter your research topic:")

    model = ChatOpenAI()
    vector_db = Chroma(embedding_function=OpenAIEmbeddings())

    class ResearchStateInput(TypedDict):
        question: str
        

    class ResearchState(TypedDict):
        question: str
        flattened_docs: list
        summary: str
        citations: list
        fact_check: str
        errors: list = []
        content: Annotated[list, operator.add]
        new_questions: list
        vector: List[Document]

    class ResearchStateOutput(TypedDict):
        flattened_docs: str
        summary: str
        vector: List[Document]
        flattened_docs: list

    def get_user_input(state: ResearchStateInput):
        return {'question': state['question']}

    def wikiloader(state: ResearchStateInput)->ResearchState:
        """To load the information from Wikipedia based on the user input question."""

        question = state['question']
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        wiki_result = wikipedia.run(question)
        doc = Document(page_content=wiki_result, metadata={"source": "Wikipedia", "title": question})
        return {'content': [doc]}

    def arxiv_loader(state: ResearchStateInput)->ResearchState:
        """To load the information from Arxiv based on the user input question."""

        question = state['question']
        arxiv = ArxivLoader(
                            query=question,
                            load_max_docs=2)
        content = arxiv.load()

        return {'content': [content]}

    def query_translation(state: ResearchStateInput)->ResearchState:
        """To translate user input queries into better queries/questions that can be used to retrieve information from the web."""

        question = state['question']
        query_translation_prompt = """You are expert at translating user input queries into a better queries/questions that 
                            can be used to retrieve information from the web. Here is your user-input question: {question}\n
                            Output (2 queries):"""

        question_template = query_translation_prompt.format(question=question)
        response = model.invoke(question_template)

        return {'new_questions': response.content.split('\n')}

    def tavily_search_docs(state: ResearchState):
        """To search the web for the information based on the translated queries/questions and get the url details."""
        
        result_doc = []
        new_questions = state['new_questions']
        web_search_prompt = """You are expert at retrieving information from the web. 
                            You are asked to find the following information: {question}"""

        tavily = TavilySearchResults(max_results=1)
        for question in new_questions:
            prompt = web_search_prompt.format(question=question)
            tavily_search = tavily.invoke({'query': prompt})
            result_doc.extend([Document(page_content=tavily['content'], metadata={"source": tavily['url'], "title": question}) 
                                for tavily in (tavily_search)])
            
        return {'content': result_doc}

    def retrieve(state: ResearchState):
        content = state['content']
        question = state['question']

        flattened_docs = [item for sublist in content for item in (sublist if isinstance(sublist, list) else [sublist])]
        vector = vector_db.add_documents(flattened_docs)

        return {'flattened_docs': flattened_docs, 'question': question, 'vector': vector}


    def summarization(state: ResearchState):
        """To summarize the retrieved information from the web."""
        flattened_docs = state['flattened_docs']
        question = state['question']

        summarization_llm = ChatOpenAI(model="gpt-4o-mini")

        summarization_prompt = """You are expert at summarizing the retrieved information from the different sources. 
                                You are asked to summarize the following information and retain the citations: {documents}"""

        summarization_prompt = summarization_prompt.format(documents=flattened_docs)

        try:
            response = model.invoke(summarization_prompt)
        except Exception as _:
            response = summarization_llm.invoke(summarization_prompt)

        citations = [doc.metadata.get('source', 'Arxiv') for doc in flattened_docs]
        return {'summary': response.content, 'citations': list(set(citations))}

    def fact_check_agent(state: ResearchState):
        """To fact-check the summarized information."""
        summary = state['summary']

        fact_check_prompt = """You are expert at fact-checking the summarized information. 
                            You are asked to fact-check the following information: {summary}"""

        fact_check_prompt = fact_check_prompt.format(summary=summary)
        response = model.invoke(fact_check_prompt)

        return {'fact_check': response.content}

    def error_detection_agent(state: ResearchState):
        """To detect the errors in the fact-checked information."""
        fact_check = state['fact_check']
        errors = []

        if "conflicting" in fact_check.lower():
            errors.append("Conflict identified in the fact-checked information.")

        return {'errors': errors}

    def error_checker(state: ResearchState)->ResearchStateOutput:
        errors = state['errors']
        
        if errors:
            return ['get_user_input']

        return END

    def rag_result(question, document, summary):
        template = """Generate answer of the given question: {question} \n based on the following summary: {summary} \n
                    Also adding full document: {document}"""

        document = "\n".join([doc.metadata.get('Summary', '') if 'Summary' in doc.metadata else doc.page_content for doc in document ])
        # prompt = prompt.format(summary=result['summary'], document=document)
        prompt = ChatPromptTemplate.from_template(template)
        # print(prompt)

        llm = ChatOpenAI(model="gpt-4o-mini")
        rag_chain =  (RunnablePassthrough().bind(
                        question=RunnablePassthrough(),
                        summary=RunnablePassthrough(),
                        document=RunnablePassthrough()
                    ) 
                    | prompt
                    | llm
                    | StrOutputParser())

        return rag_chain
            

    builder = StateGraph(ResearchStateInput, output=ResearchState)
    builder.add_node('get_user_input', get_user_input)
    builder.add_node('wikiloader', wikiloader)
    builder.add_node('arxiv_loader', arxiv_loader)
    builder.add_node('query_translation', query_translation)
    builder.add_node('tavily_search_docs', tavily_search_docs)
    builder.add_node('retrieve', retrieve)
    builder.add_node('summarization', summarization)
    builder.add_node('fact_check_agent', fact_check_agent)
    builder.add_node('error_detection_agent', error_detection_agent)

    builder.add_edge(START, 'get_user_input')
    builder.add_edge('get_user_input', 'wikiloader')
    builder.add_edge('get_user_input', 'arxiv_loader')
    builder.add_edge('get_user_input', 'query_translation')
    builder.add_edge('query_translation', 'tavily_search_docs')

    builder.add_edge('arxiv_loader', 'retrieve')
    builder.add_edge('wikiloader', 'retrieve')
    builder.add_edge('tavily_search_docs', 'retrieve')
    builder.add_edge('retrieve', 'summarization')
    builder.add_edge('summarization', 'fact_check_agent')
    builder.add_edge('fact_check_agent', 'error_detection_agent')

    builder.add_conditional_edges('error_detection_agent', error_checker, ['get_user_input', END])

    graph = builder.compile()

    if user_question:

        result = graph.invoke({'question':f'{user_question}'})
        st.subheader("Research Summary:")
        st.write(result['summary'])
        st.subheader("Citations:")
        st.write(result['citations'])
        vector = result['vector']
        followup = st.text_input('Ask any question based on the summary:')
        if followup:
            st.subheader("Follow-up Response:")
            rag_model = rag_result(followup, result['flattened_docs'], result['summary'])
            rag_model_stream = rag_model.stream({
                                'question': followup, 
                                'document': result['flattened_docs'], 
                                'summary': result['summary']
                            })

            response_placeholder = st.empty()

            full_response = ""
            for chunk in rag_model_stream:
                full_response += chunk  # Accumulate streamed content
                response_placeholder.write(full_response)  # Update Streamlit UI in real time