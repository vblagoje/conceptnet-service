import os
import sys
import json
from pathlib import Path

from app.graph_resolver import GraphResolver
from app.util import download

if __name__ == "__main__":

    cpnet_vocab_path = {"path": "./app/data/concept.txt",
                        "url": "https://storage.googleapis.com/hf_datasets/concept.txt"}
    pattern_path = {"path": "./app/data/matcher_patterns.json",
                    "url": "https://storage.googleapis.com/hf_datasets/matcher_patterns.json"}
    pruned_graph_path = {"path": "./app/data/conceptnet.en.pruned.graph",
                         "url": "https://storage.googleapis.com/hf_datasets/conceptnet.en.pruned.graph"}
    required_data = [cpnet_vocab_path, pattern_path, pruned_graph_path]

    for file in required_data:
        if not os.path.exists(file['path']):
            parent_dir = Path(file['path']).parent.absolute()
            os.makedirs(str(parent_dir), exist_ok=True)
            download(file['url'], file['path'])
    print(sys.argv[1])
    graph_resolver = GraphResolver(cpnet_vocab_path=cpnet_vocab_path['path'],
                                   pattern_path=pattern_path['path'],
                                   pruned_graph_path=pruned_graph_path['path'])
    graph_resolver.start()
    common_sense_qa_example = {"answerKey": "A", "id": "1afa02df02c908a558b4036e80242fac",
                               "question": {"question_concept": "revolving door",
                                            "choices": [{"label": "A", "text": "bank"},
                                                        {"label": "B", "text": "library"},
                                                        {"label": "C", "text": "department store"},
                                                        {"label": "D", "text": "mall"},
                                                        {"label": "E", "text": "new york"}],
                                            "stem": "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?"}}
    statement_file = open(sys.argv[1], "r")
    for line in statement_file:
        common_sense_qa_example = json.loads(line)
        statement = graph_resolver.convert_qajson_to_entailment(qa_json=common_sense_qa_example, ans_pos=False)
        grouned_statements = graph_resolver.ground(statement)
        result = graph_resolver.generate_adj_data_from_grounded_concepts__use_lm(statement, grouned_statements)

