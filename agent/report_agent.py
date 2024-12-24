from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph

from agent.build_section_with_web_research import *

# ------------------------------------------------------------
# LLMs 

# gpt_4o = ChatOpenAI(model="gpt-4o", temperature=0)

# ------------------------------------------------------------
# Search

tavily_client = TavilyClient()
tavily_async_client = AsyncTavilyClient()

# ------------------------------------------------------------
# Core graph nodes
# part 1
async def generate_report_plan(state: ReportState, config: RunnableConfig):
    """ Generate the report plan """

    # Inputs
    topic = state["topic"]
    suggested_urls_dict = state.get("suggested_urls_dict", {})

    # Get configuration
    configurable = configuration.Configuration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    tavily_topic = configurable.tavily_topic
    tavily_days = configurable.tavily_days

    # Convert JSON object to string if necessary
    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    # llm
    llm_model = configurable.llm_selector
    gpt_4o = ChatOpenAI(model=llm_model, temperature=0)

    # Generate search query
    structured_llm = gpt_4o.with_structured_output(Queries)

    # Format system instructions
    system_instructions_query = report_planner_query_writer_instructions.format(topic=topic, report_organization=report_structure, number_of_queries=number_of_queries)

    # Generate queries  
    results = structured_llm.invoke([SystemMessage(content=system_instructions_query)]+[HumanMessage(content="Generate search queries that will help with planning the sections of the report.")])

    # Web search
    query_list = [query.search_query for query in results.queries]

    # Search web 
    search_docs = await tavily_search_async(query_list, tavily_topic, tavily_days)

    # Deduplicate and format sources
    source_str = deduplicate_and_format_sources(search_docs, max_tokens_per_source=1000, include_raw_content=False)

    # Format system instructions
    system_instructions_sections = report_planner_instructions.format(topic=topic, report_organization=report_structure, context=source_str)

    # Generate sections 
    structured_llm = gpt_4o.with_structured_output(Sections)
    report_sections = structured_llm.invoke([SystemMessage(content=system_instructions_sections)]+[HumanMessage(content="Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. Each section must have: name, description, plan, research, and content fields.")])

    for section in report_sections.sections:
        #reset value from any llm data
        section.suggested_urls = []
        #check if section name is contained in any suggested sections (account for variation in LLM section titles)
        for suggested_section in suggested_urls_dict.keys():
            if section.name.lower() in suggested_section:
                #set suggested list
                section.suggested_urls = suggested_urls_dict[suggested_section]

    return {"sections": report_sections.sections}

def initiate_section_writing(state: ReportState):
    """ This is the "map" step when we kick off web research for some sections of the report """    
        
    # Kick off section writing in parallel via Send() API for any sections that require research
    return [
        Send("build_section_with_web_research", {"section": s, "exclude_domains": state.get("exclude_domains", None), "include_domains": state.get("include_domains", None)})
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
