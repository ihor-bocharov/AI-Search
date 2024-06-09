
import nest_asyncio
from llama_index.llms.openai import OpenAI
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    BaseExtractor,
)
from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline

from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.question_gen.prompts import (
    DEFAULT_SUB_QUESTION_PROMPT_TMPL,
)
from copy import deepcopy
from llama_index.core.schema import MetadataMode

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

llm_model = "gpt-4-turbo"
llm_question = "gpt-3.5-turbo-0125"

def run_pipeline():
    nest_asyncio.apply()

    uber_data = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\data\\metadata\\10k-132.pdf"
    lift_data = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\data\\metadata\\10k-vFinal.pdf"

    llm = OpenAI(temperature=0.1, model=llm_question, max_tokens=512)
    text_splitter = TokenTextSplitter(
        separator=" ", chunk_size=512, chunk_overlap=128
    )
    extractors = [
        TitleExtractor(nodes=5, llm=llm),
        QuestionsAnsweredExtractor(questions=1, llm=llm),
        EntityExtractor(prediction_threshold=0.5),
        SummaryExtractor(summaries=["prev", "self"], llm=llm),
        KeywordExtractor(keywords=5, llm=llm),
        # CustomExtractor()
    ]

    transformations = [text_splitter] + extractors

    uber_docs = SimpleDirectoryReader(input_files=[uber_data]).load_data()
    uber_front_pages = uber_docs[0:3]
    uber_content = uber_docs[63:69]
    uber_docs = uber_front_pages + uber_content

    pipeline = IngestionPipeline(transformations=transformations)
    uber_nodes = pipeline.run(documents=uber_docs)
    uber_nodes[1].metadata

    lyft_docs = SimpleDirectoryReader(
        input_files=[lift_data]
    ).load_data()
    lyft_front_pages = lyft_docs[0:3]
    lyft_content = lyft_docs[68:73]
    lyft_docs = lyft_front_pages + lyft_content
    pipeline = IngestionPipeline(transformations=transformations)
    lyft_nodes = pipeline.run(documents=lyft_docs)
    lyft_nodes[2].metadata

    question_gen = LLMQuestionGenerator.from_defaults(
        llm=llm,
        prompt_template_str="""
            Follow the example, but instead of giving a question, always prefix the question 
            with: 'By first identifying and quoting the most relevant sources, '. 
            """
        + DEFAULT_SUB_QUESTION_PROMPT_TMPL,
    )

    # Querying an Index With No Extra Metadata
    nodes_no_metadata = deepcopy(uber_nodes) + deepcopy(lyft_nodes)
    for node in nodes_no_metadata:
        node.metadata = {
            k: node.metadata[k]
            for k in node.metadata
            if k in ["page_label", "file_name"]
        }
    print(
        "LLM sees:\n",
        (nodes_no_metadata)[9].get_content(metadata_mode=MetadataMode.LLM),
    )

    index_no_metadata = VectorStoreIndex(
        nodes=nodes_no_metadata,
    )
    engine_no_metadata = index_no_metadata.as_query_engine(
        similarity_top_k=10, llm=OpenAI(model=llm_question)
    )

    final_engine_no_metadata = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[
            QueryEngineTool(
                query_engine=engine_no_metadata,
                metadata=ToolMetadata(
                    name="sec_filing_documents",
                    description="financial information on companies",
                ),
            )
        ],
        question_gen=question_gen,
        use_async=True,
    )

    response_no_metadata = final_engine_no_metadata.query(
        """
        #What was the cost due to research and development v.s. sales and marketing for uber and lyft in 2019 in millions of USD?
        #Give your answer as a JSON.
        """
    )
    print(response_no_metadata.response)


    # Querying an Index With Extracted Metadata
    print(
        "LLM sees:\n",
        (uber_nodes + lyft_nodes)[9].get_content(metadata_mode=MetadataMode.LLM),
    )

    index = VectorStoreIndex(
        nodes=uber_nodes + lyft_nodes,
    )
    engine = index.as_query_engine(similarity_top_k=10, llm=OpenAI(model=llm_model))


    final_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[
            QueryEngineTool(
                query_engine=engine,
                metadata=ToolMetadata(
                    name="sec_filing_documents",
                    description="financial information on companies.",
                ),
            )
        ],
        question_gen=question_gen,
        use_async=True,
    )


    response = final_engine.query(
        """
        What was the cost due to research and development v.s. sales and marketing for uber and lyft in 2019 in millions of USD?
        Give your answer as a JSON.
        """
    )
    print(response.response)

class CustomExtractor(BaseExtractor):
    def extract(self, nodes):
        metadata_list = [
            {
                "custom": (
                    node.metadata["document_title"]
                    + "\n"
                    + node.metadata["excerpt_keywords"]
                )
            }
            for node in nodes
        ]
        return metadata_list