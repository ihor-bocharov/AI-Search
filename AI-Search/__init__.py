from basic.pipeline import run_pipeline as run_basic_pipeline
from graph1.pipeline import run_pipeline as run_neo4j_pipeline

from dotenv import load_dotenv
import logging
import sys

def main():

    load_dotenv('../.env')
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    questions = ["Who is John McCarthy?", "Who is Sam Altman?"]

    run_basic_pipeline(questions)
    run_neo4j_pipeline(questions)

if __name__ == "__main__":
    main()