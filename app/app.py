from typing import Dict, Any

from fastapi import FastAPI
from .graph_resolver import GraphResolver

app = FastAPI()
graph_resolver = GraphResolver(cpnet_vocab_path="./app/data/concept.txt",
                               pattern_path="./app/data/matcher_patterns.json",
                               pruned_graph_path="./app/data/conceptnet.en.unpruned.graph")


def get_subgraph(common_sense_qa_example: Dict[str, str]):
    statement = graph_resolver.convert_qajson_to_entailment(qa_json=common_sense_qa_example, ans_pos=False)
    grouned_statement = graph_resolver.ground(statement)
    return graph_resolver.generate_adj_data_from_grounded_concepts__use_lm(grouned_statement)


@app.on_event("startup")
def on_startup():
    # Not needed if you setup a migration system like Alembic
    graph_resolver.start()


