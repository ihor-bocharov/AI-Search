from basic_vector.pipeline import run_pipeline as run_basic_vector_pipeline
from basic_semantic_vector.pipeline import run_pipeline as run_basic_semantic_vector_pipeline
from knowledge_graph.pipeline import run_pipeline as run_graph_pipeline
from metadata_filtering.pipeline import run_pipeline as run_metadata_pipeline
from agentic.pipeline import run_pipeline as run_agentic_pipeline

from dotenv import load_dotenv
import logging
import sys
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, ServiceContext
import llama_index.core
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
import asyncio

def main():

    load_dotenv('../.env')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    #logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    file_handler = logging.FileHandler('trace.txt')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    #llama_index.core.set_global_handler("simple")

    # Tracing
    #llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    #Settings.callback_manager = CallbackManager([llama_debug])

    #Settings.llm = OpenAI(temperature=0.0, model="gpt-4")
    Settings.llm = OpenAI(temperature=0.0, model="gpt-3.5-turbo")
    Settings.service_context = ServiceContext.from_defaults(llm=Settings.llm)

    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    Settings.context_window = 4096
    Settings.num_output = 256

    #questions = ["Who is John McCarthy?", "Who is Sam Altman?"]
    questions = ["What is the User-Agent?"]

    #run_basic_vector_pipeline(questions, load_from_storage=False)
    #run_basic_semantic_vector_pipeline(questions, load_from_storage=True)
    #run_graph_pipeline(questions, load_from_storage=True)
    #run_metadata_pipeline()
    run_agentic_pipeline(questions, load_from_storage=True)
    
if __name__ == "__main__":
    main()