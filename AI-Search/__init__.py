from basic.pipeline import run_pipeline as run_basic_pipeline
from graph1.pipeline import run_pipeline as run_neo4j_pipeline

from dotenv import load_dotenv
import logging
import sys
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

def main():

    load_dotenv('../.env')
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    Settings.llm = OpenAI(temperature=0.2, model="gpt-3.5-turbo")
    Settings.chunk_size = 512

    questions = ["Who is John McCarthy?", "Who is Sam Altman?"]

    run_basic_pipeline(questions, True)
    run_neo4j_pipeline(questions, True)

if __name__ == "__main__":
    main()