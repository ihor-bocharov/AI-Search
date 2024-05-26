import os
from pathlib import Path
import pickle
from typing import List
import nest_asyncio
import helpers.file_helper as file_helper
from  .extensions import CustomRetriever, CustomObjectRetriever

from llama_hub.file.unstructured.base import UnstructuredReader
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SummaryIndex,
    Document,
    download_loader,
    load_index_from_storage,
    StorageContext,
)
from llama_index.core.objects import (
    ObjectIndex,
    SimpleToolNodeMapping,
)
from llama_index.core.agent import ReActAgent

# Settings
doc_limit = 1314
file_list_name = "files.txt"
source_data_dir = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\data\\docs.llamaindex.ai"
base_store_path = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\persistent\\docs.llamaindex.ai"
vector_index_store_path = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\persistent\\docs.llamaindex.ai\\vector-index"
summary_index_store_path = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\persistent\\docs.llamaindex.ai\\summary-index"
summary_extracted_store_path = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\persistent\\docs.llamaindex.ai\\summary-extracted"

def run_pipeline(questions: list[str], load_from_storage: bool):
    nest_asyncio.apply()

    if not load_from_storage:
        file_helper.remove_directory_tree(base_store_path)
        docs = load_documents_from_source(source_data_dir, doc_limit)
    else:
        docs = []

    agents_dict, extra_info_dict = build_agents(docs, load_from_storage)

    # define tool for each document agent
    all_tools = []
    for file_base, agent in agents_dict.items():
        summary = extra_info_dict[file_base]["summary"]
        doc_tool = QueryEngineTool(
            query_engine=agent,
            metadata=ToolMetadata(name=f"tool_{file_base}", description=summary)
        )
        all_tools.append(doc_tool)

    tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)
    obj_index = ObjectIndex.from_objects(
            all_tools,
            tool_mapping,
            VectorStoreIndex,
        )
    vector_node_retriever = obj_index.as_node_retriever(similarity_top_k=10)

    custom_node_retriever = CustomRetriever(vector_node_retriever)

    # wrap it with ObjectRetriever to return objects
    custom_obj_retriever = CustomObjectRetriever(
        custom_node_retriever, tool_mapping, all_tools, llm=Settings.llm
    )

    #tmps = custom_obj_retriever.retrieve("hello")
    #print(len(tmps))
    top_agent = ReActAgent.from_tools(
        tool_retriever=custom_obj_retriever,
        system_prompt=""" \
    You are an agent designed to answer queries about the documentation.
    Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

    """,
        llm= Settings.llm,
        verbose=True,
    )

    print("==================================================", end='\n')
    print("Agentic RAG started", end="\n\n")
          
    for question in questions:
        print("--------------------------------------------------", end="\n\n")
        print("Q : " + question, end='\n')
        response = top_agent.query(question)
        print("A : " + str(response), end='\n')

def load_documents_from_source(data_dir: str, doc_limit: int) -> List[Document]:
    UnstructuredReader = download_loader('UnstructuredReader')
    reader = UnstructuredReader()

    all_files_gen = Path(data_dir).rglob("*")
    all_files = [f.resolve() for f in all_files_gen]
    all_html_files = [f for f in all_files if f.suffix.lower() == ".html"]
    print("==================================================", end='\n')
    print("Loaded " + str(len(all_html_files)) + " files", end='\n')

    docs = []
    files = []
    files_count = len(all_html_files)
    for idx, f in enumerate(all_html_files):
        if idx >= doc_limit:
            break
        print(f"File {idx} from {doc_limit}. Total : {files_count}")
        loaded_docs = reader.load_data(file=f, split_documents=True)

        # Hardcoded Index. Everything before this is ToC for all pages
        start_idx = 72
        loaded_doc = Document(
            text="\n\n".join([d.get_content() for d in loaded_docs[start_idx:]]),
            metadata={"path": str(f)},
        )
        print("Loaded document : " + loaded_doc.metadata["path"])
        docs.append(loaded_doc)

        files.append(create_doc_key(f))

    file_helper.save_list_to_file(files, base_store_path, file_list_name)

    return docs

def build_agents(docs: List[Document], load_from_storage: bool):
    node_parser = SentenceSplitter()

    # Build agents dictionary
    agents_dict = {}
    extra_info_dict = {}

    if not load_from_storage:
        for doc in docs:
            nodes = node_parser.get_nodes_from_documents([doc])

            # ID will be base + parent
            file_key = create_doc_key(Path(doc.metadata["path"]))
            agent, summary = create_agent_per_doc(nodes, file_key, load_from_storage)

            agents_dict[file_key] = agent
            extra_info_dict[file_key] = {"summary": summary, "nodes": nodes}
    else:
        file_keys = file_helper.load_list_from_file(base_store_path, file_list_name)
        for file_key in file_keys:
            agent, summary = create_agent_per_doc([], file_key, load_from_storage)

            agents_dict[file_key] = agent
            extra_info_dict[file_key] = {"summary": summary, "nodes": []}

    return agents_dict, extra_info_dict

def create_agent_per_doc(nodes, file_key, load_from_storage: bool):
    print("File key : " + file_key)
    
    vector_index_out_path = os.path.join(vector_index_store_path, file_key)
    summary_index_out_path = os.path.join(summary_index_store_path, file_key)
    summary_out_file = os.path.join(summary_extracted_store_path, file_key +"_summary.pkl")
    if not load_from_storage:
        # build vector index
        vector_index = VectorStoreIndex(nodes, service_context=Settings.service_context)
        vector_index.storage_context.persist(persist_dir=vector_index_out_path)

        # build summary index
        summary_index = SummaryIndex(nodes, service_context=Settings.service_context)
        summary_index.storage_context.persist(persist_dir=summary_index_out_path)
    else:
        # build vector index
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=vector_index_out_path),
            service_context=Settings.service_context
        )

        # build summary index
        summary_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=summary_index_out_path),
            service_context=Settings.service_context
        )

    # define query engines
    vector_query_engine = vector_index.as_query_engine(llm=Settings.llm)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        llm=Settings.llm,
        use_async=False
    )

    # extract a summary
    if not load_from_storage:
        Path(summary_out_file).parent.mkdir(parents=True, exist_ok=True)
        summary = str(summary_query_engine.query("Extract a concise 1-2 line summary of this document"))
        pickle.dump(summary, open(summary_out_file, "wb"))
    else:
        summary = pickle.load(open(summary_out_file, "rb"))

    agent = build_agent(vector_query_engine, summary_query_engine, file_key)
    print("Summary : " + summary, end='\n\n')

    return agent, summary

def build_agent(vector_query_engine, summary_query_engine, file_key: str):
    # define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name=f"vector_tool_{file_key}",
                description=f"Useful for questions related to specific facts",
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name=f"summary_tool_{file_key}",
                description=f"Useful for summarization questions",
            ),
        ),
    ]

    # build agent
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=Settings.llm,
        verbose=True,
        system_prompt=f"""\
You are a specialized agent designed to answer queries about the `{file_key}.html` part of the LlamaIndex docs.
You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
""",
    )

    return agent

def create_doc_key(file_path:str):
    return str(file_path.parent.stem) + "_" + str(file_path.stem)
