from basic.pipeline import run_pipeline as run_basic_pipeline
from graph1.pipeline import run_pipeline as run_neo4j_pipeline

from dotenv import load_dotenv
import logging
import sys
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
import llama_index.core

def main():

    load_dotenv('../.env')
    #logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    #logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    llama_index.core.set_global_handler("simple")

    Settings.llm = OpenAI(temperature=0.1, model="gpt-4")
    Settings.chunk_size = 512
    Settings.context_window = 4096
    Settings.num_output = 256

    questions = ["Who is John McCarthy?", "Who is Sam Altman?"]

    run_basic_pipeline(questions, True)
    run_neo4j_pipeline(questions, True)

if __name__ == "__main__":
    main()