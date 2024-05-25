import os
import shutil
from pathlib import Path
import pickle
from typing import List
import nest_asyncio
from tqdm.notebook import tqdm
from llama_hub.file.unstructured.base import UnstructuredReader
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import download_loader, SimpleDirectoryReader
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SummaryIndex,
    Document,
    load_index_from_storage,
    StorageContext,
)

# Settings
doc_limit = 10
source_data_dir = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\data\\docs.llamaindex.ai"
vector_index_store_path = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\persistent\\docs.llamaindex.ai"
summary_store_path = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\persistent\\docs.llamaindex.ai"

def run_pipeline(questions: list[str], load_from_storage: bool):
    nest_asyncio.apply()

    if not load_from_storage:
        shutil.rmtree(vector_index_store_path)

    docs = load_documents_from_source(source_data_dir, doc_limit)
    agents_dict, extra_info_dict = build_agents(docs, load_from_storage)


def load_documents_from_source(data_dir: str, doc_limit: int) -> List[Document]:
    UnstructuredReader = download_loader('UnstructuredReader')
    reader = UnstructuredReader()

    all_files_gen = Path(data_dir).rglob("*")
    all_files = [f.resolve() for f in all_files_gen]
    all_html_files = [f for f in all_files if f.suffix.lower() == ".html"]
    print("==================================================", end='\n')
    print("Loaded " + str(len(all_html_files)) + " files", end='\n')

    docs = []
    files_count = len(all_html_files)
    for idx, f in enumerate(all_html_files):
        if idx > doc_limit:
            break
        print(f"File {idx} from {files_count}")
        loaded_docs = reader.load_data(file=f, split_documents=True)

        # Hardcoded Index. Everything before this is ToC for all pages
        start_idx = 72
        loaded_doc = Document(
            text="\n\n".join([d.get_content() for d in loaded_docs[start_idx:]]),
            metadata={"path": str(f)},
        )
        print("Loaded document : " + loaded_doc.metadata["path"])
        docs.append(loaded_doc)

    return docs

def build_agents(docs, load_from_storage: bool):
    node_parser = SentenceSplitter()

    # Build agents dictionary
    agents_dict = {}
    extra_info_dict = {}

    # # this is for the baseline
    # all_nodes = []

    for idx, doc in enumerate(docs):
        nodes = node_parser.get_nodes_from_documents([doc])
        # all_nodes.extend(nodes)

        # ID will be base + parent
        file_path = Path(doc.metadata["path"])
        file_base = str(file_path.parent.stem) + "_" + str(file_path.stem)
        agent, summary = build_agent_per_doc(nodes, file_base, load_from_storage)

        agents_dict[file_base] = agent
        extra_info_dict[file_base] = {"summary": summary, "nodes": nodes}

    return agents_dict, extra_info_dict

def build_agent_per_doc(nodes, file_base, load_from_storage: bool):
    print("File base : " + file_base)
    
    vector_index_out_path = os.path.join(vector_index_store_path, file_base)
    summary_out_path = os.path.join(summary_store_path, file_base +"_summary.pkl")
    if not os.path.exists(vector_index_out_path):
        # Create root
        Path(vector_index_store_path).mkdir(parents=True, exist_ok=True)

        # build vector index
        vector_index = VectorStoreIndex(nodes, service_context=Settings.service_context)
        vector_index.storage_context.persist(persist_dir=vector_index_out_path)
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=vector_index_out_path),
            service_context=Settings.service_context,
        )

    # build summary index
    summary_index = SummaryIndex(nodes, service_context=Settings.service_context)

    # define query engines
    vector_query_engine = vector_index.as_query_engine()
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize"
    )

    # extract a summary
    if not os.path.exists(summary_out_path):
        Path(summary_out_path).parent.mkdir(parents=True, exist_ok=True)
        summary = str(summary_query_engine.aquery("Extract a concise 1-2 line summary of this document"))
        pickle.dump(summary, open(summary_out_path, "wb"))
    else:
        summary = pickle.load(open(summary_out_path, "rb"))

    # define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name=f"vector_tool_{file_base}",
                description=f"Useful for questions related to specific facts",
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name=f"summary_tool_{file_base}",
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
You are a specialized agent designed to answer queries about the `{file_base}.html` part of the LlamaIndex docs.
You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
""",
    )

    return agent, summary


