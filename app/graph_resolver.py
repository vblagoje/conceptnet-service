import json
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Union, Dict, List

import networkx as nx
import nltk
import numpy as np
import spacy
import torch
from scipy.sparse import coo_matrix
from spacy.matcher import Matcher
from transformers import RobertaForMaskedLM, AutoTokenizer

from .convert_csqa import create_hypothesis, get_fitb_from_question, create_output_dict

blacklist = ["-PRON-", "actually", "likely", "possibly", "want",
             "make", "my", "someone", "sometimes_people", "sometimes", "would", "want_to",
             "one", "something", "sometimes", "everybody", "somebody", "could", "could_be"
             ]

merged_relations = [
    'antonym',
    'atlocation',
    'capableof',
    'causes',
    'createdby',
    'isa',
    'desires',
    'hassubevent',
    'partof',
    'hascontext',
    'hasproperty',
    'madeof',
    'notcapableof',
    'notdesires',
    'receivesaction',
    'relatedto',
    'usedfor',
]

nltk.download('stopwords', quiet=True)
nltk_stopwords = nltk.corpus.stopwords.words('english')


class RobertaForMaskedLMwithLoss(RobertaForMaskedLM):

    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, masked_lm_labels=None):
        assert attention_mask is not None
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0]  # hidden_states of final layer (batch_size, sequence_length, hidden_size)
        prediction_scores = self.lm_head(sequence_output)
        outputs = (prediction_scores, sequence_output) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            bsize, seqlen = input_ids.size()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size),
                                      masked_lm_labels.view(-1)).view(bsize, seqlen)
            masked_lm_loss = (masked_lm_loss * attention_mask).sum(dim=1)
            outputs = (masked_lm_loss,) + outputs
            # (masked_lm_loss), prediction_scores, sequence_output, (hidden_states), (attentions)
        return outputs


logger = logging.getLogger(__name__)


class GraphResolver:

    def __init__(self, cpnet_vocab_path: Union[Path, str],
                 pattern_path: Union[Path, str], pruned_graph_path: Union[Path, str],
                 model_name_or_path: Union[Path, str] = "roberta-large", device: str = "cuda"):
        """
        Initialize the Grounder.
        :param cpnet_vocab_path: the path to the CPNet vocabulary
        :param pattern_path: the path to the pattern matcher
        """

        self.cpnet_vocab_path = cpnet_vocab_path
        self.pattern_path = pattern_path
        self.pruned_graph_path = pruned_graph_path

        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
        self.nlp.add_pipe('sentencizer')

        self.matcher = Matcher(vocab=self.nlp.vocab)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = RobertaForMaskedLMwithLoss.from_pretrained(model_name_or_path)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.cpnet_vocab = None
        self.cpnet_vocab_underscores = None
        self.concept2id = None
        self.id2relation = None
        self.id2concept = None
        self.relation2id = None

        self.cpnet_simple = None
        self.cpnet = None

    def start(self) -> None:
        print("Starting Graph Resolver, loading cpnet vocab...")
        with open(self.cpnet_vocab_path, "r", encoding="utf8") as f:
            file_contents = [line.strip() for line in f]
        self.cpnet_vocab = [c.replace("_", " ") for c in file_contents]
        self.cpnet_vocab_underscores = [l for l in file_contents]
        print("Loading pattern matcher...")
        with open(self.pattern_path, "r", encoding="utf8") as fin:
            all_patterns = json.load(fin)

        for concept, pattern in all_patterns.items():
            self.matcher.add(concept, [pattern])

        print("Loading pruned graph, please wait...")
        with open(self.cpnet_vocab_path, "r", encoding="utf8") as fin:
            self.id2concept = [w.strip() for w in fin]
        self.concept2id = {w: i for i, w in enumerate(self.id2concept)}

        self.id2relation = merged_relations
        self.relation2id = {r: i for i, r in enumerate(self.id2relation)}

        self.cpnet = nx.read_gpickle(self.pruned_graph_path)
        self.cpnet_simple = nx.Graph()
        for u, v, data in self.cpnet.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if self.cpnet_simple.has_edge(u, v):
                self.cpnet_simple[u][v]['weight'] += w
            else:
                self.cpnet_simple.add_edge(u, v, weight=w)
        print("Graph Resolver started.")

    def convert_qajson_to_entailment(self, qa_json: Dict, ans_pos: bool = False):
        question_text = qa_json["question"]["stem"]
        choices = qa_json["question"]["choices"]
        for choice in choices:
            choice_text = choice["text"]
            pos = None
            if not ans_pos:
                statement = create_hypothesis(get_fitb_from_question(question_text), choice_text, ans_pos)
            else:
                statement, pos = create_hypothesis(get_fitb_from_question(question_text), choice_text, ans_pos)
            create_output_dict(qa_json, statement, choice["label"] == qa_json.get("answerKey", "A"), ans_pos, pos)

        return qa_json

    def lemmatize(self, concept: str):

        doc = self.nlp(concept.replace("_", " "))
        lcs = set()
        lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
        return lcs

    def ground_mentioned_concepts(self, sentence: str, answer: str = None):

        sentence = sentence.lower()
        doc = self.nlp(sentence)
        matches = self.matcher(doc)

        mentioned_concepts = set()
        span_to_concepts = {}

        if answer is not None:
            ans_matcher = Matcher(self.nlp.vocab)
            ans_words = self.nlp(answer)
            # print(ans_words)
            ans_matcher.add(answer, [[{'TEXT': token.text.lower()} for token in ans_words]])

            ans_match = ans_matcher(doc)
            ans_mentions = set()
            for _, ans_start, ans_end in ans_match:
                ans_mentions.add((ans_start, ans_end))

        for match_id, start, end in matches:
            if answer is not None:
                if (start, end) in ans_mentions:
                    continue

            span = doc[start:end].text  # the matched span

            # a word that appears in answer is not considered as a mention in the question
            # if len(set(span.split(" ")).intersection(set(ans.split(" ")))) > 0:
            #     continue
            original_concept = self.nlp.vocab.strings[match_id]
            original_concept_set = set()
            original_concept_set.add(original_concept)

            # print("span", span)
            # print("concept", original_concept)
            # print("Matched '" + span + "' to the rule '" + string_id)

            # why do you lemmatize a mention whose len == 1?

            if len(original_concept.split("_")) == 1:
                # tag = doc[start].tag_
                # if tag in ['VBN', 'VBG']:

                original_concept_set.update(self.lemmatize(self.nlp.vocab.strings[match_id]))

            if span not in span_to_concepts:
                span_to_concepts[span] = set()

            span_to_concepts[span].update(original_concept_set)

        for span, concepts in span_to_concepts.items():
            concepts_sorted = list(concepts)
            # print("span:")
            # print(span)
            # print("concept_sorted:")
            # print(concepts_sorted)
            concepts_sorted.sort(key=len)

            # mentioned_concepts.update(concepts_sorted[0:2])

            shortest = concepts_sorted[0:3]

            for c in shortest:
                if c in blacklist:
                    continue

                # a set with one string like: set("like_apples")
                lcs = self.lemmatize(c)
                intersect = lcs.intersection(shortest)
                if len(intersect) > 0:
                    mentioned_concepts.add(list(intersect)[0])
                else:
                    mentioned_concepts.add(c)

            # if a mention exactly matches with a concept

            exact_match = set(
                [concept for concept in concepts_sorted if concept.replace("_", " ").lower() == span.lower()])
            # print("exact match:")
            # print(exact_match)
            assert len(exact_match) < 2
            mentioned_concepts.update(exact_match)
        return mentioned_concepts

    def ground_qa_pair(self, statement: str, answer: str):
        all_concepts = self.ground_mentioned_concepts(statement, answer)
        answer_concepts = self.ground_mentioned_concepts(answer)
        question_concepts = all_concepts - answer_concepts
        if len(question_concepts) == 0:
            question_concepts = self.hard_ground(statement)  # not very possible

        if len(answer_concepts) == 0:
            answer_concepts = self.hard_ground(answer)  # some case

        # question_concepts = question_concepts -  answer_concepts
        question_concepts = sorted(list(question_concepts))
        answer_concepts = sorted(list(answer_concepts))
        return {"sent": statement, "ans": answer, "qc": question_concepts, "ac": answer_concepts}

    def hard_ground(self, sent: str):
        sent = sent.lower()
        doc = self.nlp(sent)
        res = set()
        for t in doc:
            if t.lemma_ in self.cpnet_vocab:
                res.add(t.lemma_)
        sent = " ".join([t.text for t in doc])
        if sent in self.cpnet_vocab:
            res.add(sent)
        try:
            assert len(res) > 0
        except Exception:
            print(f"for {sent}, concept not found in hard grounding.")
        return res

    def match_mentioned_concepts(self, statements: List[str], answers: List[str]):
        grounded_examples: List[Dict[str]] = []
        for statement, answer in zip(statements, answers):
            grounded_examples.append(self.ground_qa_pair(statement, answer))
        return grounded_examples

    # To-do: examine prune
    def prune(self, grounded_examples: List[Dict[str, str]]):
        prune_examples = []
        for item in grounded_examples:
            qc = item["qc"]
            prune_qc = []
            for c in qc:
                if c[-2:] == "er" and c[:-2] in qc:
                    continue
                if c[-1:] == "e" and c[:-1] in qc:
                    continue
                have_stop = False
                # remove all concepts having stopwords, including hard-grounded ones
                for t in c.split("_"):
                    if t in nltk_stopwords:
                        have_stop = True
                if not have_stop and c in self.cpnet_vocab_underscores:
                    prune_qc.append(c)

            ac = item["ac"]
            prune_ac = []
            for c in ac:
                if c[-2:] == "er" and c[:-2] in ac:
                    continue
                if c[-1:] == "e" and c[:-1] in ac:
                    continue
                all_stop = True
                for t in c.split("_"):
                    if t not in nltk_stopwords:
                        all_stop = False
                if not all_stop and c in self.cpnet_vocab_underscores:
                    prune_ac.append(c)

            try:
                assert len(prune_ac) > 0 and len(prune_qc) > 0
            except Exception as e:
                pass
                # print("In pruning")
                # print(prune_qc)
                # print(prune_ac)
                # print("original:")
                # print(qc)
                # print(ac)
                # print()
            item["qc"] = prune_qc
            item["ac"] = prune_ac

            prune_examples.append(item)
        return prune_examples

    def ground(self, common_sense_example: Dict) -> Dict:

        # common_sense_example: Dict
        """
        {'answerKey': 'B',
         'id': 'b8c0a4703079cf661d7261a60a1bcbff',
         'question': {'question_concept': 'magazines',
                      'choices': [{'label': 'A', 'text': 'doctor'}, {'label': 'B', 'text': 'bookstore'},
                                  {'label': 'C', 'text': 'market'}, {'label': 'D', 'text': 'train station'},
                                  {'label': 'E', 'text': 'mortuary'}],
                      'stem': 'Where would you find magazines along side many other printed works?'},
         'statements': [
             {'label': False, 'statement': 'Doctor would you find magazines along side many other printed works.'},
             {'label': True, 'statement': 'Bookstore would you find magazines along side many other printed works.'},
             {'label': False, 'statement': 'Market would you find magazines along side many other printed works.'},
             {'label': False,
              'statement': 'Train station would you find magazines along side many other printed works.'},
             {'label': False, 'statement': 'Mortuary would you find magazines along side many other printed works.'}]}
        """
        statements = []
        answers = []

        for statement in common_sense_example["statements"]:
            statements.append(statement["statement"])

        for answer in common_sense_example["question"]["choices"]:
            ans = answer['text']
            try:
                assert all([i != "_" for i in ans])
            except Exception:
                print(ans)
            answers.append(ans)

        grounded_concepts = self.match_mentioned_concepts(statements, answers)
        res = self.prune(grounded_concepts)

        return res

    def concepts2adj(self, node_ids):
        cids = np.array(node_ids, dtype=np.int32)
        n_rel = len(self.id2relation)
        n_node = cids.shape[0]
        adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
        for s in range(n_node):
            for t in range(n_node):
                s_c, t_c = cids[s], cids[t]
                if self.cpnet.has_edge(s_c, t_c):
                    for e_attr in self.cpnet[s_c][t_c].values():
                        if 0 <= e_attr['rel'] < n_rel:
                            adj[e_attr['rel']][s][t] = 1
        # cids += 1  # note!!! index 0 is reserved for padding
        adj = coo_matrix(adj.reshape(-1, n_node))
        return adj, cids

    def get_lm_score(self, cids, question):
        cids = cids[:]
        cids.insert(0, -1)  # QAcontext node
        sents, scores = [], []
        for cid in cids:
            if cid == -1:
                sent = question.lower()
            else:
                sent = '{} {}.'.format(question.lower(), ' '.join(self.id2concept[cid].split('_')))
            sent = self.tokenizer.encode(sent, add_special_tokens=True)
            sents.append(sent)
        n_cids = len(cids)
        cur_idx = 0
        batch_size = 50
        while cur_idx < n_cids:
            # Prepare batch
            input_ids = sents[cur_idx: cur_idx + batch_size]
            max_len = max([len(seq) for seq in input_ids])
            for j, seq in enumerate(input_ids):
                seq += [self.tokenizer.pad_token_id] * (max_len - len(seq))
                input_ids[j] = seq
            input_ids = torch.tensor(input_ids).to(self.device)  # [B, seqlen]
            mask = (input_ids != 1).long()  # [B, seq_len]
            # Get LM score
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=mask, masked_lm_labels=input_ids)
                loss = outputs[0]  # [B, ]
                _scores = list(-loss.detach().cpu().numpy())  # list of float
            scores += _scores
            cur_idx += batch_size
        assert len(sents) == len(scores) == len(cids)
        cid2score = OrderedDict(sorted(list(zip(cids, scores)), key=lambda x: -x[1]))  # score: from high to low
        return cid2score

    def concepts_to_adj_matrices_2hop_all_pair__use_lm__part1(self, data):
        qc_ids, ac_ids, question = data
        qa_nodes = set(qc_ids) | set(ac_ids)
        extra_nodes = set()
        for qid in qa_nodes:
            for aid in qa_nodes:
                if qid != aid and qid in self.cpnet_simple.nodes and aid in self.cpnet_simple.nodes:
                    extra_nodes |= set(self.cpnet_simple[qid]) & set(self.cpnet_simple[aid])
        extra_nodes = extra_nodes - qa_nodes
        return sorted(qc_ids), sorted(ac_ids), question, sorted(extra_nodes)

    def concepts_to_adj_matrices_2hop_all_pair__use_lm__part2(self, data):
        qc_ids, ac_ids, question, extra_nodes = data
        cid2score = self.get_lm_score(qc_ids + ac_ids + extra_nodes, question)
        return qc_ids, ac_ids, question, extra_nodes, cid2score

    def concepts_to_adj_matrices_2hop_all_pair__use_lm__part3(self, data):
        qc_ids, ac_ids, question, extra_nodes, cid2score = data
        schema_graph = qc_ids + ac_ids + sorted(extra_nodes, key=lambda x: -cid2score[x])  # score: from high to low
        arange = np.arange(len(schema_graph))
        qmask = arange < len(qc_ids)
        amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
        adj, concepts = self.concepts2adj(schema_graph)
        return {'adj': [i.tolist() for i in adj.nonzero()],'adj_shape':[adj.shape[0], adj.shape[1]],
                'concepts': concepts.tolist(),
                'qmask': qmask.tolist(), 'amask': amask.tolist(),
                'cid2score': [[i[0], i[1].item()] for i in cid2score.items()]}

    def generate_adj_data_from_grounded_concepts__use_lm(self, statement, grounded_statements):
        """
        This function will save
            (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
            (2) concepts ids
            (3) qmask that specifices whether a node is a question concept
            (4) amask that specifices whether a node is a answer concept
            (5) cid2score that maps a concept id to its relevance score given the QA context
        to the output path in python pickle format

        grounded_path: str
        cpnet_graph_path: str
        cpnet_vocab_path: str
        output_path: str
        """

        qa_data = []
        for grounded_statement in grounded_statements:
            q_ids = set(self.concept2id[c] for c in grounded_statement['qc'])
            a_ids = set(self.concept2id[c] for c in grounded_statement['ac'])
            q_ids = q_ids - a_ids
            qa_context = "{} {}.".format(statement['question']['stem'], grounded_statement['ans'])
            qa_data.append((q_ids, a_ids, qa_context))

        result = []
        for qa_item in qa_data:
            qa_item_adj = self.concepts_to_adj_matrices_2hop_all_pair__use_lm__part1(qa_item)
            qa_item_adj_scored = self.concepts_to_adj_matrices_2hop_all_pair__use_lm__part2(qa_item_adj)
            final_item = self.concepts_to_adj_matrices_2hop_all_pair__use_lm__part3(qa_item_adj_scored)
            result.append(final_item)
        return result
