from basic_vector.pipeline import run_pipeline as run_basic_vector_pipeline
from basic_semantic_vector.pipeline import run_pipeline as run_basic_semantic_vector_pipeline
from knowledge_graph.pipeline import run_pipeline as run_graph_pipeline
from metadata_filtering.pipeline import run_pipeline as run_metadata_pipeline

from dotenv import load_dotenv
import logging
import sys
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
import llama_index.core

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

    #Settings.llm = OpenAI(temperature=0.0, model="gpt-4")
    Settings.llm = OpenAI(temperature=0.0, model="gpt-3.5-turbo")
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    Settings.context_window = 4096
    Settings.num_output = 256

    questions = ["Who is John McCarthy?", "Who is Sam Altman?"]

    #run_basic_vector_pipeline(questions, load_from_storage=False)
    #run_basic_semantic_vector_pipeline(questions, load_from_storage=True)
    #run_graph_pipeline(questions, load_from_storage=True)
    run_metadata_pipeline()

if __name__ == "__main__":
    main()