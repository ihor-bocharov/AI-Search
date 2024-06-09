import os
import logging
from pathlib import Path
import pickle
from typing import List
import nest_asyncio
import helpers.file_helper as file_helper
import helpers.display_helper as display_helper
from  .extensions import CustomRetriever, CustomObjectRetriever
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
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
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, DatasetGenerator
from llama_index.core.node_parser import SentenceSplitter

# Settings
doc_limit = 10
file_list_name = "files.txt"
file_questions_name = "Questions.txt"

source_data_dir = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\data\\docs.llamaindex.ai"
base_store_path = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\persistent\\docs.llamaindex.ai"
basic_vector_index_store_path = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\persistent\\docs.llamaindex.ai\\basic-vector-index"
vector_index_store_path = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\persistent\\docs.llamaindex.ai\\vector-index"
summary_index_store_path = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\persistent\\docs.llamaindex.ai\\summary-index"
summary_extracted_store_path = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\persistent\\docs.llamaindex.ai\\summary-extracted"

#evaluation_model = "gpt-4-turbo"
evaluation_model = "gpt-3.5-turbo"

evaluation_llm = OpenAI(temperature=0.0, model=evaluation_model)

faithfulness_evaluator = FaithfulnessEvaluator(llm=evaluation_llm)
relevancy_evaluator = RelevancyEvaluator(llm=evaluation_llm)

def run_pipeline(questions: list[str], load_from_storage: bool, token_counter):
    nest_asyncio.apply()

    logging.info("Evaluation Model : " + evaluation_model)

    documents  = []
    if not load_from_storage:
        file_helper.remove_directory_tree(base_store_path)
        documents = load_documents_from_source(source_data_dir, doc_limit)

    agents_dict, extra_info_dict = build_agents(documents, load_from_storage, token_counter)

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

    top_agent = ReActAgent.from_tools(
        tool_retriever=custom_obj_retriever,
        system_prompt=""" \
    You are an agent designed to answer queries about the documentation.
    Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

    """,
        llm= Settings.llm,
        verbose=True,
    )
    print("|============================================================")
    print("|Agentic RAG started")
    logging.info("|============================================================")
    logging.info("|Agentic RAG started")

    agentic_faithfulness_list = []
    agentic_relevancy_list = []
    for question in questions:
        print("|------------------------------------------------------------")
        logging.info("|------------------------------------------------------------")
        q = "|Q : " + question
        print(q)
        logging.info(q)

        response = top_agent.query(question)

        answer = "|A : " + str(response)
        print(answer)
        logging.info(answer)

        print("|Faithfulness Evaluation")
        logging.info("|Faithfulness Evaluation")
        faithfulness_eval_result = faithfulness_evaluator.evaluate_response(query=question, response=response)
        agentic_faithfulness_list.append({"question": question, "response": response, "eval_result": faithfulness_eval_result})
        display_helper.display_evaluation_result(faithfulness_eval_result)

        print("|Relevancy Evaluation")
        logging.info("|Relevancy Evaluation")
        relevancy_eval_result = relevancy_evaluator.evaluate_response(query=question, response=response)
        agentic_relevancy_list.append({"question": question, "response": response, "eval_result": relevancy_eval_result})
        display_helper.display_evaluation_result(relevancy_eval_result)

        print("|End Evaluation")
        logging.info("|End Evaluation")

    # Basic RAG
    if load_from_storage:
        basic_vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=basic_vector_index_store_path),
            service_context=Settings.service_context
        )
    else:
        basic_vector_index = VectorStoreIndex.from_documents(documents, callback_manager=Settings.callback_manager)
        basic_vector_index.storage_context.persist(basic_vector_index_store_path)

    basic_query_engine = basic_vector_index.as_query_engine(similarity_top_k=4)

    print("||============================================================")
    print("||Basic RAG started")
    logging.info("||============================================================")
    logging.info("||Basic RAG started")

    basic_faithfulness_list = []
    basic_relevancy_list = []
    for question in questions:
        print("||------------------------------------------------------------")
        logging.info("||------------------------------------------------------------")
    
        q = "||Q : " + question
        print(q)
        logging.info(q)

        response = basic_query_engine.query(question)

        answer = "||A : " + str(response)
        print(answer)
        logging.info(answer)

        print("||Faithfulness Evaluation")
        logging.info("||Faithfulness Evaluation")
        faithfulness_eval_result = faithfulness_evaluator.evaluate_response(query=question, response=response)
        basic_faithfulness_list.append({"question": question, "response": response, "eval_result": faithfulness_eval_result})
        display_helper.display_evaluation_result(faithfulness_eval_result)

        print("||Relevancy Evaluation")
        logging.info("||Relevancy Evaluation")
        relevancy_eval_result = relevancy_evaluator.evaluate_response(query=question, response=response)
        basic_relevancy_list.append({"question": question, "response": response, "eval_result": relevancy_eval_result})
        display_helper.display_evaluation_result(relevancy_eval_result)
        
        print("||End Evaluation")
        logging.info("||End Evaluation")

    print("============================================================")
    logging.info("============================================================")

    score = display_helper.calculate_results_score(agentic_faithfulness_list)
    score_message = f'Agentic Faithfulness score: {score:.4f}'
    print(score_message)
    logging.info(score_message)

    score = display_helper.calculate_results_score(agentic_relevancy_list)
    score_message = f'Agentic Relevance score: {score:.4f}'
    print(score_message)
    logging.info(score_message)

    score = display_helper.calculate_results_score(basic_faithfulness_list)
    score_message = f'Basic Faithfulness score: {score:.4f}'
    print(score_message)
    logging.info(score_message)

    score = display_helper.calculate_results_score(basic_relevancy_list)
    score_message = f'Basic Relevance score: {score:.4f}'
    print(score_message)
    logging.info(score_message)

def load_documents_from_source(data_dir: str, doc_limit: int) -> List[Document]:
    UnstructuredReader = download_loader('UnstructuredReader')
    reader = UnstructuredReader()

    all_files_gen = Path(data_dir).rglob("*")
    all_files = [f.resolve() for f in all_files_gen]
    all_html_files = [f for f in all_files if f.suffix.lower() == ".html"]
    print("==================================================")
    print("Loaded " + str(len(all_html_files)) + " files")

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

def build_agents(docs: List[Document], load_from_storage: bool, token_counter):
    # Sentence Splitter
    node_parser = SentenceSplitter()

    # Build agents dictionary
    agents_dict = {}
    extra_info_dict = {}

    if not load_from_storage:
        print("Start index creating")
        logging.info("Start index creating")
    
        for doc in docs:
            nodes = node_parser.get_nodes_from_documents([doc])

            # ID will be base + parent
            file_key = create_doc_key(Path(doc.metadata["path"]))
            agent, summary = create_agent_per_doc(nodes, file_key, load_from_storage, token_counter)

            agents_dict[file_key] = agent
            extra_info_dict[file_key] = {"summary": summary, "nodes": nodes}

        print("End index creating")
        logging.info("End index creating")
    else:
        print("Start index loading")
        logging.info("Start index loading")
        file_keys = file_helper.load_list_from_file(base_store_path, file_list_name)
        for file_key in file_keys:
            agent, summary = create_agent_per_doc([], file_key, load_from_storage, token_counter)

            agents_dict[file_key] = agent
            extra_info_dict[file_key] = {"summary": summary, "nodes": []}

        print("End index loading")
        logging.info("End index loading")
    return agents_dict, extra_info_dict

def create_agent_per_doc(nodes, file_key, load_from_storage: bool, token_counter):
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
    #print("Summary : " + summary, end='\n\n')

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

def create_basic_index():
    documents = load_documents_from_source(source_data_dir, doc_limit)
    
    # Basic
    index = VectorStoreIndex.from_documents(documents, callback_manager=Settings.callback_manager)

    # Store index
    index.storage_context.persist(basic_vector_index_store_path)

    