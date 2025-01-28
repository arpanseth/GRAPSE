import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from dotenv import load_dotenv
# Load the environment variables used for evaluation
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from argparse import ArgumentParser
from openai import OpenAI

from classes import Pipeline

__all__ = ["MainPipeline"]

class MainPipeline(Pipeline):
    def setup(self):
        self.client = OpenAI()

    def get_answer(self, question):
        client = self.client
        response = client.chat.completions.create(
        model = self.model_name,
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": question,
                }
            ],
            }
        ],
        )
        return response.choices[0].message.content

if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("-q", "--question", type=str, required=True, help="Question to ask the model")
    parser.add_argument("-m", "--model", type=str, required=False, default="gpt-4o-mini", help="Model to use for the answer")
    args = parser.parse_args()
    pipeline = MainPipeline(args.model)
    pipeline.setup()
    print(pipeline.get_answer(args.question))