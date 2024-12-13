# -- Imports
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import agent.configuration as configuration
from agent.utils import *


gpt_4o = ChatOpenAI(model="gpt-4o", temperature=0)


# ------------------------------------------------------------
# Graph nodes

def generate_queries(state: SectionState, config: RunnableConfig):
    """ Generate search queries for a report section """

    # Get state
    section = state["section"]

    # Get configuration
    configurable = configuration.Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # Generate queries
    structured_llm = gpt_4o.with_structured_output(Queries)

    # Format system instructions
    system_instructions = query_writer_instructions.format(section_topic=section.description,
                                                           number_of_queries=number_of_queries)

    # Generate queries
    queries = structured_llm.invoke([SystemMessage(content=system_instructions)] + [
        HumanMessage(content="Generate search queries on the provided topic.")])

    return {"search_queries": queries.queries}


async def search_web(state: SectionState, config: RunnableConfig):
    """ Search the web for each query, then return a list of raw sources and a formatted string of sources."""

    # Get state
    search_queries = state["search_queries"]

    # Get configuration
    configurable = configuration.Configuration.from_runnable_config(config)
    tavily_topic = configurable.tavily_topic
    tavily_days = configurable.tavily_days

    # Web search
    query_list = [query.search_query for query in search_queries]
    search_docs = await tavily_search_async(query_list, tavily_topic, tavily_days)

    # Deduplicate and format sources
    source_str = deduplicate_and_format_sources(search_docs, max_tokens_per_source=5000, include_raw_content=True)

    return {"source_str": source_str}


def write_section(state: SectionState):
    """ Write a section of the report """

    # Get state
    section = state["section"]
    source_str = state["source_str"]

    # Format system instructions
    system_instructions = section_writer_instructions.format(section_title=section.name,
                                                             section_topic=section.description, context=source_str)

    # Generate section
    section_content = gpt_4o.invoke([SystemMessage(content=system_instructions)] + [
        HumanMessage(content="Generate a report section based on the provided sources.")])

    # Write content to the section object
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}


