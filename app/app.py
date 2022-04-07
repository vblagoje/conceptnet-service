import os
from pathlib import Path

from fastapi import FastAPI, Request

from .graph_resolver import GraphResolver
from .util import download

cpnet_vocab_path = {"path": "./app/data/concept.txt", "url": "https://storage.googleapis.com/hf_datasets/concept.txt"}
pattern_path = {"path": "./app/data/matcher_patterns.json",
                "url": "https://storage.googleapis.com/hf_datasets/matcher_patterns.json"}
pruned_graph_path = {"path": "./app/data/conceptnet.en.pruned.graph",
                     "url": "https://storage.googleapis.com/hf_datasets/conceptnet.en.pruned.graph"}

app = FastAPI()

required_data = [cpnet_vocab_path, pattern_path, pruned_graph_path]

for file in required_data:
    if not os.path.exists(file['path']):
        parent_dir = Path(file['path']).parent.absolute()
        os.makedirs(str(parent_dir), exist_ok=True)
        download(file['url'], file['path'])

graph_resolver = GraphResolver(cpnet_vocab_path=cpnet_vocab_path['path'],
                               pattern_path=pattern_path['path'],
                               pruned_graph_path=pruned_graph_path['path'])


@app.post("/csqa/")
async def resolve_csqa(request: Request):
    # Waits for the request and converts into JSON
    common_sense_qa_example = await request.json()
    statement = graph_resolver.convert_qajson_to_entailment(qa_json=common_sense_qa_example, ans_pos=False)
    grouned_statements = graph_resolver.ground(statement)
    result = graph_resolver.generate_adj_data_from_grounded_concepts__use_lm(statement, grouned_statements)


@app.on_event("startup")
def on_startup():
    # Not needed if you setup a migration system like Alembic
    graph_resolver.start()
