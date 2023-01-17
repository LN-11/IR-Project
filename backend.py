from nltk.corpus import stopwords
from nltk.stem.porter import *
import numpy as np
from inverted_index_gcp import *
import math
import builtins

corpus_size = 6348910
TUPLE_SIZE = 6
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

# collecting all stopwords
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
all_stopwords = set(english_stopwords).union(corpus_stopwords)


def tokenize(text):
    """
    Tokenize a string into a list of tokens.
    """
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    tokens_filtered = [term for term in tokens if term not in all_stopwords]
    return tokens_filtered


def get_pl(inverted, term, index_dir, th=500):
    """
    Get posting list of a term.
    """
    with closing(MultiFileReader()) as reader:
        # setting the threshold
        th = min(inverted.df[term], th)
        locs = inverted.posting_locs[term]
        # reading the posting list
        b = reader.read(locs, th * TUPLE_SIZE, index_dir)
        posting_list = []
        # decoding the posting list
        for i in range(th):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


def get_cands(query, index, index_dir, th=400):
    """
    Get candidate documents for a query.
    """
    cands = set()
    term_pls = {}
    # get posting lists for each term
    for term in set(query):
        if term in index.df.keys():
            pl = get_pl(index, term, index_dir, th)
            term_pls[term] = dict(pl)
            cands.update([doc for doc, _ in term_pls[term].items()])
    return cands, term_pls


def get_top_n(sim_dict, N=5000):
    """
    Get top N documents from a similarity dictionary.
    """
    return dict(sorted(sim_dict.items(), key=lambda k: k[1], reverse=True)[:N])


def tf_idf(doc_id, query, inverted, word_dict):
    """
    Calculate tf-idf score for a document.
    """
    score = 0
    token_count = Counter(query)
    query_vec = []
    if doc_id not in inverted.dl: return 0
    # calculate tf-idf score
    for token in token_count:
        if token in inverted.df and doc_id in word_dict[token]:
            # tf
            doc_tf = word_dict[token][doc_id] / inverted.dl[doc_id]
            doc_idf = math.log10(corpus_size / inverted.df[token])
            query_tf = token_count[token] / len(query)
            query_idf = doc_idf
            score += doc_tf * doc_idf * query_tf * query_idf
            query_vec.append((query_tf * query_idf) ** 2)
    # calculate cosine similarity
    query_norm = math.sqrt(sum(query_vec))
    docs_norm = inverted.d_norms[doc_id]
    return score / (query_norm * docs_norm)


def combine_scores(rel_docs, title_matches, bm_score, cosim_score, pr_score, pv_score):
    """
    Combine scores from different models.
    """
    # calculate the final score
    did_score = {rel_doc: np.array([0] * 5, dtype=float) for rel_doc in rel_docs}
    measures = [title_matches, bm_score, cosim_score, pr_score, pv_score]
    # combine scores
    for doc_id in rel_docs:
        for i, measure in enumerate(measures):
            if len(measure) != 0 and max(measure.values()) != 0:
                did_score[doc_id][i] = measure.get(doc_id, 0) / max(measure.values()) - min(measure.values())
    return did_score


def get_matches(query, term_pls, doc_id):
    """
    Get number of matched terms in a document.
    """
    for term in query:
        if term not in term_pls.keys() or \
                doc_id not in term_pls[term].keys():
            return 0
    return 1


class BM25_from_index:
    """
    Calculate BM25 score for a document.
    """

    def __init__(self, index, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(index.dl)
        self.AVGDL = builtins.sum(index.dl.values()) / self.N

    def calc_idf(self, list_of_tokens):
        """
        Calculate idf for a list of tokens.
        """
        idf = {}
        # calculate idf for each term
        for term in list_of_tokens:
            if term in self.index.df:
                term_df = self.index.df[term]
                idf[term] = math.log(1 + (self.N - term_df + 0.5) / (term_df + 0.5))
        return idf

    def search(self, query, relevant_docs, word_dict, N=5000):
        """
        Search for a query.
        """
        self.idf = self.calc_idf(set([t for t in query]))
        # calculate BM25 score for each document
        return get_top_n(
            dict([(relevant_doc, self._score(query, relevant_doc, word_dict)) for relevant_doc in relevant_docs]), N)

    def _score(self, query, doc_id, word_dict):
        """
        Calculate BM25 score for a document.
        """
        score = 0.0
        if doc_id == 0.0:
            return 0
        doc_len = self.index.dl[doc_id]
        for term in query:
            if term in self.index.df:
                term_frequencies = word_dict[term]
                if doc_id in term_frequencies:
                    freq = term_frequencies[doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score
