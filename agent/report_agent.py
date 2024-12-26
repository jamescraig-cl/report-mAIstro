from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from agent.build_section_with_web_research import *

# ------------------------------------------------------------
# Search

tavily_client = TavilyClient()
tavily_async_client = AsyncTavilyClient()

# ------------------------------------------------------------
# Core graph nodes
# part 1
async def generate_report_plan(state: ReportState, config: RunnableConfig):
    """ Skip generation and pass section JSON as Pydantic Sections to next nodes """

    report_sections = state["report_specification"].get("sections")
    namespace = state["report_specification"].get("company").replace(" ", "")
    print('set namespace ' + namespace)
    # Get configuration
    configurable = configuration.Configuration.from_runnable_config(config)

    pc_index = configurable.pc_index

    #Serialize as Pydantic Section
    pydantic_sections = [Section(**section) for section in report_sections]

    if state.get("scrape_indicator", ""):
        #collect list of urls to scrape
        scrape_urls = [s.get("suggested_urls", []) for s in report_sections]
        flat_scrape_urls = [url for section_list in scrape_urls for url in section_list]

        embedding_model = "text-embedding-3-large"
        embeddings = OpenAIEmbeddings(model=embedding_model)
        for url in flat_scrape_urls:
            #scrape each url and return formatted/cleaned chunks
            docs = await scrape_async(url)
            #upsert to Pinecone
            PineconeVectorStore.from_documents(
                docs,
                index_name=pc_index,
                embedding=embeddings,
                namespace=namespace
            )

    return {"sections": pydantic_sections, "rag_namespace": namespace}

def initiate_section_writing(state: ReportState):
    """ This is the "map" step when we kick off web research for some sections of the report """    
        
    # Kick off section writing in parallel via Send() API for any sections that require research
    return [
        Send("build_section_with_web_research", {"section": s, "exclude_domains": state.get("exclude_domains", None), "include_domains": state.get("include_domains", None), "rag_namespace": state.get("rag_namespace", "")})
        for s in state["sections"] 
        if s.research
    ]

# part 2

def gather_completed_sections(state: ReportState):
    """ Gather completed sections from research and format them as context for writing the final sections """

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(completed_sections)

    return {"report_sections_from_research": completed_report_sections}

#intro, conclusion, and other non-research sections
def initiate_final_section_writing(state: ReportState):
    """ Write any final sections using the Send API to parallelize the process """    

    # Kick off section writing in parallel via Send() API for any sections that do not require research
    return [
        Send("write_final_sections", {"section": s, "report_sections_from_research": state["report_sections_from_research"]}) 
        for s in state["sections"] 
        if not s.research
    ]

#intro, conclusion, and other non-research sections
def write_final_sections(state: SectionState, config: RunnableConfig):
    """ Write final sections of the report, which do not require web search and use the completed sections as context """

    # Get state
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]

    # Get configuration
    configurable = configuration.Configuration.from_runnable_config(config)

    # llm
    llm_model = configurable.llm_selector
    gpt_4o = ChatOpenAI(model=llm_model, temperature=0)

    # Format system instructions
    system_instructions = final_section_writer_instructions.format(section_title=section.name,
                                                                   section_topic=section.description,
                                                                   context=completed_report_sections)
    # Generate section
    section_content = gpt_4o.invoke([SystemMessage(content=system_instructions)] + [
        HumanMessage(content="Generate a report section based on the provided sources.")])

    # Write content to section
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}

def compile_final_report(state: ReportState, config: RunnableConfig):
    """ Compile the final report """    

    # Get sections
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}

    # Get configuration
    configurable = configuration.Configuration.from_runnable_config(config)

    # llm
    llm_model = configurable.llm_selector
    gpt_4o = ChatOpenAI(model=llm_model, temperature=0)

    #att: take completed sections and conform to pydantic model
    structured_llm = gpt_4o.with_structured_output(FinalSections)
    system_instructions_json = report_compiler_json_instructions.format(context=completed_sections)
    structured_prompt = [SystemMessage(content=system_instructions_json)]+[HumanMessage(content="Return a JSON representation of the report, placing each report section into an appropriate JSON object. Using the report as context, categorize the company as a manufacturer, service provide, distributor, or other high-level company type. ")]
    #type: FinalSections
    structured_result = structured_llm.invoke(structured_prompt)

    # Update sections with completed content while maintaining original order
    for section in sections:
        section.content = completed_sections[section.name]

    # Compile final report
    all_sections = "\n\n".join([s.content for s in sections])

    return {"final_report": all_sections, "final_report_dict": structured_result}




#research sub-graph
# Add nodes and edges
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
# opt 1 RAG
# section_builder.add_node("retrieve", retrieve)
# opt 2 web
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")
section_builder.add_edge("write_section", END)

# Add nodes and edges 
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=configuration.Configuration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)
builder.add_edge(START, "generate_report_plan")
builder.add_conditional_edges("generate_report_plan", initiate_section_writing, ["build_section_with_web_research"])
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

graph = builder.compile()
