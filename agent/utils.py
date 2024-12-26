
import asyncio
import operator

from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from typing_extensions import TypedDict
from typing import  Annotated, List
from pydantic import BaseModel, Field
from langsmith import traceable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.firecrawl import FireCrawlLoader

from enum import Enum
from tavily import TavilyClient, AsyncTavilyClient

# ------------------------------------------------------------
# Search

tavily_client = TavilyClient()
tavily_async_client = AsyncTavilyClient()

# ------------------------------------------------------------
# Schema

class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web research for this section of the report."
    )
    content: str = Field(
        description="The content of the section."
    )
    suggested_urls: list[str] = Field(
        description="List of user-supplied URLs to use for web research"
    )



class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )
class SectionEnum(str, Enum):
    products = 'products'
    services = 'services'
    leaders = 'leaders'
    description = 'description'
    headquarters = 'headquarters'
    markets = 'markets'
    line_of_business = 'line_of_business'
    locations = 'locations'
    distributors = 'distributors'
    introduction = 'introduction'
    conclusion = 'conclusion'

class CategoryEnum(str, Enum):
    manufacturer = 'manufacturer'
    contract_manufacturer = 'contract manufacturer'
    services = 'services'
    distributor = 'distributor'
    software = 'software'
    media = 'media'

class FinalSection(BaseModel):
    # name: str = Field(description="The name of the section")
    name: SectionEnum
    content: str = Field(description="the content of the section")

class FinalSections(BaseModel):
    category: CategoryEnum
    finalSections: List[FinalSection] = Field(
        description="Final output sections of the report"
    )




# ------------------------------------------------------------
# State

class ReportStateInput(TypedDict):
    topic: str # Report topic
    suggested_urls_dict: dict
    scrape_indicator: bool
    include_domains: list[str]
    exclude_domains: list[str]
    report_specification: dict

class ReportStateOutput(TypedDict):
    final_report: str # Final report
    #att 1 at semi-structured output
    final_report_dict: FinalSections

class ReportState(TypedDict):
    topic: str # Report topic
    sections: list[Section] # List of report sections
    completed_sections: Annotated[list, operator.add] # Send() API key
    report_sections_from_research: str # String of any completed sections from research to write final sections
    final_report: str # Final report
    suggested_urls_dict: dict
    scrape_indicator: bool
    rag_namespace: str
    include_domains: list[str]
    exclude_domains: list[str]
    report_specification: dict


class SectionState(TypedDict):
    section: Section # Report section
    search_queries: list[SearchQuery] # List of search queries
    source_str: str # String of formatted source content from web search
    report_sections_from_research: str # String of any completed sections from research to write final sections
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API
    suggested_urls: list[str]
    rag_namespace: str
    include_domains: list[str]
    exclude_domains: list[str]

class SectionOutputState(TypedDict):
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API

# ------------------------------------------------------------
# Prompts

# Prompt to generate a search query to help with planning the report outline
report_planner_query_writer_instructions="""You are an expert technical writer, helping to plan a report. 

The report will be focused on the following topic:

{topic}

The report structure will follow these guidelines:

{report_organization}

Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information for planning the report sections. 

The query should:

1. Be related to the topic 
2. Help satisfy the requirements specified in the report organization

Make the query specific enough to find high-quality, relevant sources while covering the breadth needed for the report structure."""

# Prompt generating the report outline
report_planner_instructions="""You are an expert technical writer, helping to plan a report.

Your goal is to generate the outline of the sections of the report. 

The overall topic of the report is:

{topic}

The report should follow this organization: 

{report_organization}

You should reflect on this information to plan the sections of the report: 

{context}

Now, generate the sections of the report. Each section should have the following fields:

- Name - Name for this section of the report.
- Description - Brief overview of the main topics and concepts to be covered in this section.
- Research - Whether to perform web research for this section of the report.
- Content - The content of the section, which you will leave blank for now.

Consider which sections require web research. For example, introduction and conclusion will not require research because they will distill information from other parts of the report."""

# Query writer instructions
query_writer_instructions="""Your goal is to generate targeted web search queries that will gather comprehensive information for writing a technical report section.

Topic for this section:
{section_topic}

When generating {number_of_queries} search queries, ensure they:
1. Cover different aspects of the topic (e.g., core features, real-world applications, technical architecture)
2. Include specific technical terms related to the topic
3. Target recent information by including year markers where relevant (e.g., "2024")
4. Look for comparisons or differentiators from similar technologies/approaches
5. Search for both official documentation and practical implementation examples

Your queries should be:
- Specific enough to avoid generic results
- Technical enough to capture detailed implementation information
- Diverse enough to cover all aspects of the section plan
- Focused on authoritative sources (documentation, technical blogs, academic papers)"""

# Section writer instructions
section_writer_instructions = """You are an expert technical writer crafting one section of a technical report.

Topic for this section:
{section_topic}

Guidelines for writing:

1. Technical Accuracy:
- Include specific version numbers
- Reference concrete metrics/benchmarks
- Cite official documentation
- Use technical terminology precisely

2. Length and Style:
- Strict 150-200 word limit for paragraph-centered sections
- List- based sections should be very concise but contain as many list items as are found in the context
- No marketing language
- Technical focus
- Write in simple, clear language
- Start with your most important insight in **bold**
- Use short paragraphs (2-3 sentences max)

3. Structure:
- Use ## for section title (Markdown format)
- Only use ONE structural element IF it helps clarify your point:
  * Either a focused table comparing 2-3 key items (using Markdown table syntax)
  * Or a short list (3-5 items) using proper Markdown list syntax:
    - Use `*` or `-` for unordered lists
    - Use `1.` for ordered lists
    - Ensure proper indentation and spacing
- End with ### Sources that references the below source material formatted as:
  * List each source with title, date, and URL
  * Format: `- Title : URL`

3. Writing Approach:
- Include at least one specific example or case study
- Use concrete details over general statements
- Make every word count
- No preamble prior to creating the section content
- Focus on your single most important point

4. Use this source material to help write the section:
{context}

5. Quality Checks:

- Careful use of only ONE structural element (table or list) and only if it helps clarify your point
- One specific example / case study
- Starts with bold insight
- No preamble prior to creating the section content
- Sources cited at end"""

# dropped quality checks
# - Exactly 150-200 words (excluding title and sources)

final_section_writer_instructions="""You are an expert technical writer crafting a section that synthesizes information from the rest of the report.

Section to write: 
{section_topic}

Available report content:
{context}

1. Section-Specific Approach:

For Introduction:
- Use # for report title (Markdown format)
- 50-100 word limit
- Write in simple and clear language
- Focus on the core motivation for the report in 1-2 paragraphs
- Use a clear narrative arc to introduce the report
- Include NO structural elements (no lists or tables)
- No sources section needed

For Conclusion/Summary:
- Use ## for section title (Markdown format)
- 100-150 word limit
- For comparative reports:
    * Must include a focused comparison table using Markdown table syntax
    * Table should distill insights from the report
    * Keep table entries clear and concise
- For non-comparative reports: 
    * Only use ONE structural element IF it helps distill the points made in the report:
    * Either a focused table comparing items present in the report (using Markdown table syntax)
    * Or a short list using proper Markdown list syntax:
      - Use `*` or `-` for unordered lists
      - Use `1.` for ordered lists
      - Ensure proper indentation and spacing
- End with specific next steps or implications
- No sources section needed

3. Writing Approach:
- Use concrete details over general statements
- Make every word count
- Focus on your single most important point

4. Quality Checks:
- For introduction: 50-100 word limit, # for report title, no structural elements, no sources section
- For conclusion: 100-150 word limit, ## for section title, only ONE structural element at most, no sources section
- Markdown format
- Do not include word count or any preamble in your response"""

report_compiler_json_instructions = """
Translate the following report into JSON, placing each section in the appropriate JSON object

{context}
"""
# ------------------------------------------------------------
# Utility functions

def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=True):
    """
    Takes either a single search response or list of responses from Tavily API and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.

    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results

    Returns:
        str: Formatted string with deduplicated sources
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response['results']
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                sources_list.extend(response['results'])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")

    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        #     if source['url'] not in unique_sources:
        unique_sources[source['url']] = source

    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"

    return formatted_text.strip()


def format_sections(sections: list[Section]) -> str:
    """ Format a list of sections into a string """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
{'=' * 60}
Section {idx}: {section.name}
{'=' * 60}
Description:
{section.description}
Requires Research: 
{section.research}

Content:
{section.content if section.content else '[Not yet written]'}

"""
    return formatted_str


@traceable
def tavily_search(query):
    """ Search the web using the Tavily API.

    Args:
        query (str): The search query to execute

    Returns:
        dict: Tavily search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available"""

    return tavily_client.search(query,
                                max_results=7,
                                include_raw_content=True
                                , search_depth="advanced")


@traceable
async def tavily_search_async(search_queries, tavily_topic, tavily_days, **kwargs):
    """
    Performs concurrent web searches using the Tavily API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process
        tavily_topic (str): Type of search to perform ('news' or 'general')
        tavily_days (int): Number of days to look back for news articles (only used when tavily_topic='news')

    Returns:
        List[dict]: List of search results from Tavily API, one per query

    Note:
        For news searches, each result will include articles from the last `tavily_days` days.
        For general searches, the time range is unrestricted.
    """

    search_tasks = []

    exclude_domains = kwargs.get('exclude_domains', None)
    if exclude_domains:
        print('exclude ' + str(exclude_domains))

    include_domains = kwargs.get('include_domains', None)
    if include_domains:
        print('include ' + str(include_domains))

    for query in search_queries:
        if tavily_topic == "news":
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=7,
                    include_raw_content=True,
                    topic="news",
                    days=tavily_days,
                    search_depth="advanced",
                )
            )
        else:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=7,
                    include_raw_content=True,
                    topic="general",
                    search_depth="advanced",
                    exclude_domains=exclude_domains,
                    include_domains=include_domains
                )
            )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)

    return search_docs

@traceable
async def scrape_async(url):

    loader = FireCrawlLoader(
        url=url,
        mode="scrape",
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128,
        length_function=len,
        is_separator_regex=False,
    )
    filtered_documents = filter(lambda raw_document: "error" not in raw_document.metadata.keys(), documents)

    docs = text_splitter.split_documents(filtered_documents)

    for document in docs:
        header_string = "Title: " + document.metadata.get("title", '') + " Description: " + document.metadata.get(
            "description", '')
        document.page_content = header_string + ' ' + document.page_content

    return docs


def create_retriever(namespace, index_name):

    embedding_model = "text-embedding-3-large"
    embeddings = OpenAIEmbeddings(model=embedding_model)

    #prepare retriever tool and select by namespace
    vectorsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
    print("   ---RETRIEVER FOR '" + namespace + "'---")

    retriever = vectorsearch.as_retriever(search_kwargs={"k": 2, "namespace": namespace})

    retriever_tool = create_retriever_tool(
        retriever,
        "search_company_website",
        "Searches and returns content from within the company website.",
    )
    return retriever