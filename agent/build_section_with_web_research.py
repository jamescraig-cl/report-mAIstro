# -- Imports
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import agent.configuration as configuration
from agent.utils import *
from pinecone import Pinecone

from agent.utils import JsonFormatter

# ------------------------------------------------------------
# Logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = JsonFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)


# ------------------------------------------------------------
# Graph nodes

def generate_queries(state: SectionState, config: RunnableConfig):
    """ Generate search queries for a report section """

    # Get state
    section = state["section"]

    # Get configuration
    configurable = configuration.Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    #llm
    llm_model = configurable.llm_selector
    llm = ChatOpenAI(model=llm_model, temperature=0)

    # Generate queries
    structured_llm = llm.with_structured_output(Queries)

    # Format system instructions
    system_instructions = query_writer_instructions.format(section_topic=section.description,
                                                           number_of_queries=number_of_queries)

    # Generate queries
    queries = structured_llm.invoke([SystemMessage(content=system_instructions)] + [
        HumanMessage(content="Generate search queries on the provided topic.")])

    return {"search_queries": queries.queries}


async def gather_sources(state: SectionState, config: RunnableConfig):
    """
    Attempt RAG on any suggested sources
    Attempt web search if no suggested sources or if RAG output was poor
    Return a list of raw sources and a formatted string of sources.
    """

    # Get state
    search_queries = state["search_queries"]
    exclude_domains = state["exclude_domains"]
    rag_indicator = len(state["section"].suggested_urls)
    section_name = state["section"].name

    # Get configuration
    configurable = configuration.Configuration.from_runnable_config(config)
    tavily_topic = configurable.tavily_topic
    tavily_days = configurable.tavily_days
    retriever_k = configurable.rag_k
    rerank_n = configurable.rerank_n
    rag_threshold = configurable.rag_threshold

    # queries for web search (or RAG)
    query_list = [query.search_query for query in search_queries]
    source_str = ''

    #retrieval search
    if rag_indicator:
        logger.info("RAG search for section " + state["section"].name)
        index_name = configurable.pc_index
        retriever = create_retriever(state["rag_namespace"], index_name, retriever_k)
        rag_docs = set()
        pc = Pinecone()

        #execute all searches against RAG
        for query in query_list:
            initial_rag_docs = retriever.invoke(query)
            initial_content = [doc.page_content for doc in initial_rag_docs]
            #TODO decide whether to rerank vs each query and return top n results for each OR rerank vs section desc and keep top n of all
            logger.info('reranking ' + str(len(initial_content)) + ' docs - ic type ' + str(type(initial_content)))
            reranked_results = pc.inference.rerank(
                model="bge-reranker-v2-m3",
                query=query,
                documents=initial_content,
                top_n=rerank_n,
                return_documents=True,
            )
            reranked_docs = [{"score": doc["score"], "content": doc["document"]["text"]} for doc in reranked_results.data]
            logger.info('reranked docs - received ' + str(len(reranked_docs)) + ' results')
            for doc in reranked_docs:
                #filter results by score => soft grading
                if doc.get("content", "") not in rag_docs and doc.get("score",0) >= rag_threshold:
                    logger.info('retrieved doc for section ' + section_name + ' met quality threshold with a score of ' + str(doc.get("score")))
                    rag_docs.add(doc.get("content"))
                if doc.get("score", 0) < rag_threshold:
                    #set indicator to use web search
                    logger.info('retrieved doc for section ' + section_name + ' failed to meet quality threshold with a score of ' + str(doc.get("score")))

        #if fewer than 4 docs result from all queries, do web search
        if len(rag_docs) < 4:
            #warn that RAG for query failed
            logger.info('RAG for query "' + query + '" in section ' + section_name + ' failed. Attempting web search')
            rag_indicator = False

        #combine output from RAG, regardless of source count
        source_str = ' \n '.join(rag_docs)

    # web search if no RAG or RAG quality was poor (supplement any good RAG documents)
    if not rag_indicator:
        logger.info('Web search for section ' + state["section"].name)
        search_docs = await tavily_search_async(query_list, tavily_topic, tavily_days,
                                                exclude_domains=exclude_domains)

        # Deduplicate and format sources while keeping any approved RAG results
        source_str += ' \n ' + deduplicate_and_format_sources(search_docs, max_tokens_per_source=5000,
                                                    include_raw_content=True)

    return {"source_str": source_str}


def write_section(state: SectionState, config: RunnableConfig):
    """ Write a section of the report """

    configurable = configuration.Configuration.from_runnable_config(config)

    # Get state
    section = state["section"]
    source_str = state["source_str"]

    # Format system instructions
    system_instructions = section_writer_instructions.format(section_title=section.name, section_sources=section.suggested_urls,
                                                             section_topic=section.description, context=source_str)
    # llm
    llm_model = configurable.llm_selector
    llm = ChatOpenAI(model=llm_model, temperature=0)

    # Generate section
    section_content = llm.invoke([SystemMessage(content=system_instructions)] + [
        HumanMessage(content="Generate a report section based on the provided sources.")])

    # Write content to the section object
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}


