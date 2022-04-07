from typing import Dict, Any

from fastapi import FastAPI, Request
from .graph_resolver import GraphResolver

app = FastAPI()
graph_resolver = GraphResolver(cpnet_vocab_path="./app/data/concept.txt",
                               pattern_path="./app/data/matcher_patterns.json",
                               pruned_graph_path="./app/data/conceptnet.en.pruned.graph")


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
