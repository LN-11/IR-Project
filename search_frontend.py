from flask import Flask, request, jsonify
from backend import *
from google.cloud import storage
import pickle
import numpy as np
import pandas as pd


# Connecting to google storage bucket.
client = storage.Client()
bucket = client.get_bucket('ln3250')
title_idx = pickle.loads(bucket.get_blob('index_title_idx.pkl').download_as_string())
body_idx = pickle.loads(bucket.get_blob('index_body_idx.pkl').download_as_string())
anchor_idx = pickle.loads(bucket.get_blob('index_anchor_idx.pkl').download_as_string())
page_views = pickle.loads(bucket.get_blob('page_views.pkl').download_as_string())
page_rank = pickle.loads(bucket.get_blob('page_rank.pkl').download_as_string())
id2title = pickle.loads(bucket.get_blob('id2title.pkl').download_as_string())
bm25 = BM25_from_index(body_idx)


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    """ Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = tokenize(query)
    body_cands, body_cands_dict = get_cands(query, body_idx, "body_idx")
    title_cands, title_cands_dict = get_cands(query, title_idx, "title_idx")
    title_cands = title_cands.intersection(body_cands)

    # filter out the candidates that do not have the query terms in the title
    title_cands_updated = {}
    for term, pls in title_cands_dict.items():
        for doc_id in pls.keys():
            if doc_id in body_cands:
                title_cands_updated[term] = pls

    title_cands_dict = title_cands_updated

    title_sim_dict = {doc_id: get_matches(query, title_cands_dict, doc_id) for doc_id in title_cands}
    title_matches = get_top_n(title_sim_dict)

    bm_score = bm25.search(query, body_cands, body_cands_dict)

    cosim_dict = {doc_id: tf_idf(doc_id, query, body_idx, body_cands_dict) for doc_id in body_cands}
    cosim_score = get_top_n(cosim_dict)

    relevant_union = set().union(*[body_cands, title_cands])

    pagerank_score = get_top_n(dict([(doc_id, page_rank.get(doc_id, 0)) for doc_id in relevant_union]))
    pageview_score = get_top_n(dict([(doc_id, page_views.get(doc_id, 0)) for doc_id in relevant_union]))

    # combine all scores
    combined_scores = combine_scores(relevant_union, title_matches, bm_score, cosim_score, pagerank_score, pageview_score)
    ws = np.array([1, 4, 3, 3, 2])
    if len(query) < 3:
        ws = np.array([3, 3, 2, 3, 2])
    # return norm_scores
    top_d = sorted((combined_scores.keys()), key=lambda x: np.dot(combined_scores[x], ws), reverse=True)[:5]
    res = [(doc_id, id2title.loc[doc_id]) for doc_id in top_d if doc_id in id2title.index]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    """ Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = tokenize(query)
    cands, cands_dict = get_cands(query, body_idx, "body_idx", th=500)
    cosim_dict = {doc_id: tf_idf(doc_id, query, body_idx, cands_dict) for doc_id in cands}
    top_d = [doc_id for doc_id, _ in get_top_n(cosim_dict, N=100).items()]
    res = [(doc_id, id2title.loc[doc_id]) for doc_id in top_d if doc_id in id2title.index]
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = tokenize(query)
    cands, cands_dict = get_cands(query, title_idx, "title_idx", th=corpus_size)
    sim_dict = {doc_id: get_matches(query, cands_dict, doc_id) for doc_id in cands}
    top_d = [doc_id for doc_id, _ in get_top_n(sim_dict, N=corpus_size).items()]
    res = [(doc_id, id2title.loc[doc_id]) for doc_id in top_d if doc_id in id2title.index]
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = tokenize(query)
    cands, cands_dict = get_cands(query, anchor_idx, "anchor_idx", th=corpus_size)
    sim_dict = {doc_id: get_matches(query, cands_dict, doc_id) for doc_id in cands}
    top_d = [doc_id for doc_id, _ in get_top_n(sim_dict, N=corpus_size).items()]
    res = [(doc_id, id2title.loc[doc_id]) for doc_id in top_d if doc_id in id2title.index]
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    """ Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correspond to the provided article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = [page_rank[wiki_id] for wiki_id in wiki_ids]
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    """ Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = [page_views.get(doc_id, 0) for doc_id in wiki_ids]
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)


""" Testing we have done on the search function """
# @app.route("/search1")
# def search1():
#     """ Returns up to a 100 of your best search results for the query. This is
#         the place to put forward your best search engine, and you are free to
#         implement the retrieval whoever you'd like within the bound of the
#         project requirements (efficiency, quality, etc.). That means it is up to
#         you to decide on whether to use stemming, remove stopwords, use
#         PageRank, query expansion, etc.
#
#         To issue a query navigate to a URL like:
#          http://YOUR_SERVER_DOMAIN/search?query=hello+world
#         where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
#         if you're using ngrok on Colab or your external IP on GCP.
#     Returns:
#     --------
#         list of up to 100 search results, ordered from best to worst where each
#         element is a tuple (wiki_id, title).
#     """
#     res = []
#     query = request.args.get('query', '')
#     if len(query) == 0:
#         return jsonify(res)
#     # BEGIN SOLUTION
#     query = tokenize(query)
#     body_cands, body_cands_dict = get_cands(query, body_idx, "body_idx")
#     title_cands, title_cands_dict = get_cands(query, title_idx, "title_idx")
#     title_cands = title_cands.intersection(body_cands)
#
#     # filter out the candidates that do not have the query terms in the title
#     title_cands_updated = {}
#     for term, pls in title_cands_dict.items():
#         for doc_id in pls.keys():
#             if doc_id in body_cands:
#                 title_cands_updated[term] = pls
#
#     title_cands_dict = title_cands_updated
#
#     title_sim_dict = {doc_id: get_matches(query, title_cands_dict, doc_id) for doc_id in title_cands}
#     title_matches = get_top_n(title_sim_dict)
#
#     bm_score = bm25.search(query, body_cands, body_cands_dict)
#
#     cosim_dict = {doc_id: tf_idf(doc_id, query, body_idx, body_cands_dict) for doc_id in body_cands}
#     cosim_score = get_top_n(cosim_dict)
#
#     relevant_union = set().union(*[body_cands, title_cands])
#
#     pagerank_score = get_top_n(dict([(doc_id, page_rank.get(doc_id, 0)) for doc_id in relevant_union]))
#     pageview_score = get_top_n(dict([(doc_id, page_views.get(doc_id, 0)) for doc_id in relevant_union]))
#
#     # combine all scores
#     combined_scores = combine_scores(relevant_union, title_matches, bm_score, cosim_score, pagerank_score, pageview_score)
#     ws = np.array([1, 4, 3, 3, 2])
#     if len(query) < 3:
#         ws = np.array([3, 3, 2, 3, 2])
#     # return norm_scores
#     top_d = sorted((combined_scores.keys()), key=lambda x: np.dot(combined_scores[x], ws), reverse=True)[:20]
#     res = [(doc_id, id2title.loc[doc_id]) for doc_id in top_d if doc_id in id2title.index]
#
#     # END SOLUTION
#     return jsonify(res)
#
# @app.route("/search2")
# def search2():
#     """ Returns up to a 100 of your best search results for the query. This is
#         the place to put forward your best search engine, and you are free to
#         implement the retrieval whoever you'd like within the bound of the
#         project requirements (efficiency, quality, etc.). That means it is up to
#         you to decide on whether to use stemming, remove stopwords, use
#         PageRank, query expansion, etc.
#
#         To issue a query navigate to a URL like:
#          http://YOUR_SERVER_DOMAIN/search?query=hello+world
#         where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
#         if you're using ngrok on Colab or your external IP on GCP.
#     Returns:
#     --------
#         list of up to 100 search results, ordered from best to worst where each
#         element is a tuple (wiki_id, title).
#     """
#     res = []
#     query = request.args.get('query', '')
#     if len(query) == 0:
#         return jsonify(res)
#     # BEGIN SOLUTION
#     query = tokenize(query)
#     body_cands, body_cands_dict = get_cands(query, body_idx, "body_idx")
#     title_cands, title_cands_dict = get_cands(query, title_idx, "title_idx")
#     title_cands = title_cands.intersection(body_cands)
#
#     # filter out the candidates that do not have the query terms in the title
#     title_cands_updated = {}
#     for term, pls in title_cands_dict.items():
#         for doc_id in pls.keys():
#             if doc_id in body_cands:
#                 title_cands_updated[term] = pls
#
#     title_cands_dict = title_cands_updated
#
#     title_sim_dict = {doc_id: get_matches(query, title_cands_dict, doc_id) for doc_id in title_cands}
#     title_matches = get_top_n(title_sim_dict)
#
#     bm_score = bm25.search(query, body_cands, body_cands_dict)
#
#     cosim_dict = {doc_id: tf_idf(doc_id, query, body_idx, body_cands_dict) for doc_id in body_cands}
#     cosim_score = get_top_n(cosim_dict)
#
#     relevant_union = set().union(*[body_cands, title_cands])
#
#     pagerank_score = get_top_n(dict([(doc_id, page_rank.get(doc_id, 0)) for doc_id in relevant_union]))
#     pageview_score = get_top_n(dict([(doc_id, page_views.get(doc_id, 0)) for doc_id in relevant_union]))
#
#     # combine all scores
#     combined_scores = combine_scores(relevant_union, title_matches, bm_score, cosim_score, pagerank_score, pageview_score)
#     ws = np.array([1, 4, 3, 3, 2])
#     if len(query) < 3:
#         ws = np.array([3, 3, 2, 3, 2])
#     # return norm_scores
#     top_d = sorted((combined_scores.keys()), key=lambda x: np.dot(combined_scores[x], ws), reverse=True)[:15]
#     res = [(doc_id, id2title.loc[doc_id]) for doc_id in top_d if doc_id in id2title.index]
#
#     # END SOLUTION
#     return jsonify(res)
#
#
#
# @app.route("/search3")
# def search3():
#     """ Returns up to a 100 of your best search results for the query. This is
#         the place to put forward your best search engine, and you are free to
#         implement the retrieval whoever you'd like within the bound of the
#         project requirements (efficiency, quality, etc.). That means it is up to
#         you to decide on whether to use stemming, remove stopwords, use
#         PageRank, query expansion, etc.
#
#         To issue a query navigate to a URL like:
#          http://YOUR_SERVER_DOMAIN/search?query=hello+world
#         where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
#         if you're using ngrok on Colab or your external IP on GCP.
#     Returns:
#     --------
#         list of up to 100 search results, ordered from best to worst where each
#         element is a tuple (wiki_id, title).
#     """
#     res = []
#     query = request.args.get('query', '')
#     if len(query) == 0:
#         return jsonify(res)
#     # BEGIN SOLUTION
#     query = tokenize(query)
#     body_cands, body_cands_dict = get_cands(query, body_idx, "body_idx")
#     title_cands, title_cands_dict = get_cands(query, title_idx, "title_idx")
#     title_cands = title_cands.intersection(body_cands)
#
#     # filter out the candidates that do not have the query terms in the title
#     title_cands_updated = {}
#     for term, pls in title_cands_dict.items():
#         for doc_id in pls.keys():
#             if doc_id in body_cands:
#                 title_cands_updated[term] = pls
#
#     title_cands_dict = title_cands_updated
#
#     title_sim_dict = {doc_id: get_matches(query, title_cands_dict, doc_id) for doc_id in title_cands}
#     title_matches = get_top_n(title_sim_dict)
#
#     bm_score = bm25.search(query, body_cands, body_cands_dict)
#
#     cosim_dict = {doc_id: tf_idf(doc_id, query, body_idx, body_cands_dict) for doc_id in body_cands}
#     cosim_score = get_top_n(cosim_dict)
#
#     relevant_union = set().union(*[body_cands, title_cands])
#
#     pagerank_score = get_top_n(dict([(doc_id, page_rank.get(doc_id, 0)) for doc_id in relevant_union]))
#     pageview_score = get_top_n(dict([(doc_id, page_views.get(doc_id, 0)) for doc_id in relevant_union]))
#
#     # combine all scores
#     combined_scores = combine_scores(relevant_union, title_matches, bm_score, cosim_score, pagerank_score, pageview_score)
#     ws = np.array([1, 4, 3, 3, 2])
#     if len(query) < 3:
#         ws = np.array([3, 3, 2, 3, 2])
#     # return norm_scores
#     top_d = sorted((combined_scores.keys()), key=lambda x: np.dot(combined_scores[x], ws), reverse=True)[:12]
#     res = [(doc_id, id2title.loc[doc_id]) for doc_id in top_d if doc_id in id2title.index]
#
#     # END SOLUTION
#     return jsonify(res)
#
# @app.route("/search4")
# def search4():
#     """ Returns up to a 100 of your best search results for the query. This is
#         the place to put forward your best search engine, and you are free to
#         implement the retrieval whoever you'd like within the bound of the
#         project requirements (efficiency, quality, etc.). That means it is up to
#         you to decide on whether to use stemming, remove stopwords, use
#         PageRank, query expansion, etc.
#
#         To issue a query navigate to a URL like:
#          http://YOUR_SERVER_DOMAIN/search?query=hello+world
#         where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
#         if you're using ngrok on Colab or your external IP on GCP.
#     Returns:
#     --------
#         list of up to 100 search results, ordered from best to worst where each
#         element is a tuple (wiki_id, title).
#     """
#     res = []
#     query = request.args.get('query', '')
#     if len(query) == 0:
#         return jsonify(res)
#     # BEGIN SOLUTION
#     query = tokenize(query)
#     body_cands, body_cands_dict = get_cands(query, body_idx, "body_idx")
#     title_cands, title_cands_dict = get_cands(query, title_idx, "title_idx")
#     title_cands = title_cands.intersection(body_cands)
#
#     # filter out the candidates that do not have the query terms in the title
#     title_cands_updated = {}
#     for term, pls in title_cands_dict.items():
#         for doc_id in pls.keys():
#             if doc_id in body_cands:
#                 title_cands_updated[term] = pls
#
#     title_cands_dict = title_cands_updated
#
#     title_sim_dict = {doc_id: get_matches(query, title_cands_dict, doc_id) for doc_id in title_cands}
#     title_matches = get_top_n(title_sim_dict)
#
#     bm_score = bm25.search(query, body_cands, body_cands_dict)
#
#     cosim_dict = {doc_id: tf_idf(doc_id, query, body_idx, body_cands_dict) for doc_id in body_cands}
#     cosim_score = get_top_n(cosim_dict)
#
#     relevant_union = set().union(*[body_cands, title_cands])
#
#     pagerank_score = get_top_n(dict([(doc_id, page_rank.get(doc_id, 0)) for doc_id in relevant_union]))
#     pageview_score = get_top_n(dict([(doc_id, page_views.get(doc_id, 0)) for doc_id in relevant_union]))
#
#     # combine all scores
#     combined_scores = combine_scores(relevant_union, title_matches, bm_score, cosim_score, pagerank_score, pageview_score)
#     ws = np.array([1, 4, 3, 3, 2])
#     if len(query) < 3:
#         ws = np.array([3, 3, 2, 3, 2])
#     # return norm_scores
#     top_d = sorted((combined_scores.keys()), key=lambda x: np.dot(combined_scores[x], ws), reverse=True)[:10]
#     res = [(doc_id, id2title.loc[doc_id]) for doc_id in top_d if doc_id in id2title.index]
#
#     # END SOLUTION
#     return jsonify(res)
#
# @app.route("/search5")
# def search5():
#     """ Returns up to a 100 of your best search results for the query. This is
#         the place to put forward your best search engine, and you are free to
#         implement the retrieval whoever you'd like within the bound of the
#         project requirements (efficiency, quality, etc.). That means it is up to
#         you to decide on whether to use stemming, remove stopwords, use
#         PageRank, query expansion, etc.
#
#         To issue a query navigate to a URL like:
#          http://YOUR_SERVER_DOMAIN/search?query=hello+world
#         where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
#         if you're using ngrok on Colab or your external IP on GCP.
#     Returns:
#     --------
#         list of up to 100 search results, ordered from best to worst where each
#         element is a tuple (wiki_id, title).
#     """
#     res = []
#     query = request.args.get('query', '')
#     if len(query) == 0:
#         return jsonify(res)
#     # BEGIN SOLUTION
#     query = tokenize(query)
#     body_cands, body_cands_dict = get_cands(query, body_idx, "body_idx")
#     title_cands, title_cands_dict = get_cands(query, title_idx, "title_idx")
#     title_cands = title_cands.intersection(body_cands)
#
#     # filter out the candidates that do not have the query terms in the title
#     title_cands_updated = {}
#     for term, pls in title_cands_dict.items():
#         for doc_id in pls.keys():
#             if doc_id in body_cands:
#                 title_cands_updated[term] = pls
#
#     title_cands_dict = title_cands_updated
#
#     title_sim_dict = {doc_id: get_matches(query, title_cands_dict, doc_id) for doc_id in title_cands}
#     title_matches = get_top_n(title_sim_dict)
#
#     bm_score = bm25.search(query, body_cands, body_cands_dict)
#
#     cosim_dict = {doc_id: tf_idf(doc_id, query, body_idx, body_cands_dict) for doc_id in body_cands}
#     cosim_score = get_top_n(cosim_dict)
#
#     relevant_union = set().union(*[body_cands, title_cands])
#
#     pagerank_score = get_top_n(dict([(doc_id, page_rank.get(doc_id, 0)) for doc_id in relevant_union]))
#     pageview_score = get_top_n(dict([(doc_id, page_views.get(doc_id, 0)) for doc_id in relevant_union]))
#
#     # combine all scores
#     combined_scores = combine_scores(relevant_union, title_matches, bm_score, cosim_score, pagerank_score, pageview_score)
#     ws = np.array([1, 4, 3, 3, 2])
#     if len(query) < 3:
#         ws = np.array([3, 3, 2, 3, 2])
#     # return norm_scores
#     top_d = sorted((combined_scores.keys()), key=lambda x: np.dot(combined_scores[x], ws), reverse=True)[:8]
#     res = [(doc_id, id2title.loc[doc_id]) for doc_id in top_d if doc_id in id2title.index]
#
#     # END SOLUTION
#     return jsonify(res)
#
#
# @app.route("/search6")
# def search6():
#     """ Returns up to a 100 of your best search results for the query. This is
#         the place to put forward your best search engine, and you are free to
#         implement the retrieval whoever you'd like within the bound of the
#         project requirements (efficiency, quality, etc.). That means it is up to
#         you to decide on whether to use stemming, remove stopwords, use
#         PageRank, query expansion, etc.
#
#         To issue a query navigate to a URL like:
#          http://YOUR_SERVER_DOMAIN/search?query=hello+world
#         where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
#         if you're using ngrok on Colab or your external IP on GCP.
#     Returns:
#     --------
#         list of up to 100 search results, ordered from best to worst where each
#         element is a tuple (wiki_id, title).
#     """
#     res = []
#     query = request.args.get('query', '')
#     if len(query) == 0:
#         return jsonify(res)
#     # BEGIN SOLUTION
#     query = tokenize(query)
#     body_cands, body_cands_dict = get_cands(query, body_idx, "body_idx")
#     title_cands, title_cands_dict = get_cands(query, title_idx, "title_idx")
#     title_cands = title_cands.intersection(body_cands)
#
#     # filter out the candidates that do not have the query terms in the title
#     title_cands_updated = {}
#     for term, pls in title_cands_dict.items():
#         for doc_id in pls.keys():
#             if doc_id in body_cands:
#                 title_cands_updated[term] = pls
#
#     title_cands_dict = title_cands_updated
#
#     title_sim_dict = {doc_id: get_matches(query, title_cands_dict, doc_id) for doc_id in title_cands}
#     title_matches = get_top_n(title_sim_dict)
#
#     bm_score = bm25.search(query, body_cands, body_cands_dict)
#
#     cosim_dict = {doc_id: tf_idf(doc_id, query, body_idx, body_cands_dict) for doc_id in body_cands}
#     cosim_score = get_top_n(cosim_dict)
#
#     relevant_union = set().union(*[body_cands, title_cands])
#
#     pagerank_score = get_top_n(dict([(doc_id, page_rank.get(doc_id, 0)) for doc_id in relevant_union]))
#     pageview_score = get_top_n(dict([(doc_id, page_views.get(doc_id, 0)) for doc_id in relevant_union]))
#
#     # combine all scores
#     combined_scores = combine_scores(relevant_union, title_matches, bm_score, cosim_score, pagerank_score, pageview_score)
#     ws = np.array([1, 4, 3, 3, 2])
#     if len(query) < 3:
#         ws = np.array([3, 3, 2, 3, 2])
#     # return norm_scores
#     top_d = sorted((combined_scores.keys()), key=lambda x: np.dot(combined_scores[x], ws), reverse=True)[:5]
#     res = [(doc_id, id2title.loc[doc_id]) for doc_id in top_d if doc_id in id2title.index]
#
#     # END SOLUTION
#     return jsonify(res)
#
# @app.route("/search7")
# def search7():
#     """ Returns up to a 100 of your best search results for the query. This is
#         the place to put forward your best search engine, and you are free to
#         implement the retrieval whoever you'd like within the bound of the
#         project requirements (efficiency, quality, etc.). That means it is up to
#         you to decide on whether to use stemming, remove stopwords, use
#         PageRank, query expansion, etc.
#
#         To issue a query navigate to a URL like:
#          http://YOUR_SERVER_DOMAIN/search?query=hello+world
#         where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
#         if you're using ngrok on Colab or your external IP on GCP.
#     Returns:
#     --------
#         list of up to 100 search results, ordered from best to worst where each
#         element is a tuple (wiki_id, title).
#     """
#     res = []
#     query = request.args.get('query', '')
#     if len(query) == 0:
#         return jsonify(res)
#     # BEGIN SOLUTION
#     query = tokenize(query)
#     body_cands, body_cands_dict = get_cands(query, body_idx, "body_idx")
#     title_cands, title_cands_dict = get_cands(query, title_idx, "title_idx")
#     title_cands = title_cands.intersection(body_cands)
#
#     # filter out the candidates that do not have the query terms in the title
#     title_cands_updated = {}
#     for term, pls in title_cands_dict.items():
#         for doc_id in pls.keys():
#             if doc_id in body_cands:
#                 title_cands_updated[term] = pls
#
#     title_cands_dict = title_cands_updated
#
#     title_sim_dict = {doc_id: get_matches(query, title_cands_dict, doc_id) for doc_id in title_cands}
#     title_matches = get_top_n(title_sim_dict)
#
#     bm_score = bm25.search(query, body_cands, body_cands_dict)
#
#     cosim_dict = {doc_id: tf_idf(doc_id, query, body_idx, body_cands_dict) for doc_id in body_cands}
#     cosim_score = get_top_n(cosim_dict)
#
#     relevant_union = set().union(*[body_cands, title_cands])
#
#     pagerank_score = get_top_n(dict([(doc_id, page_rank.get(doc_id, 0)) for doc_id in relevant_union]))
#     pageview_score = get_top_n(dict([(doc_id, page_views.get(doc_id, 0)) for doc_id in relevant_union]))
#
#     # combine all scores
#     combined_scores = combine_scores(relevant_union, title_matches, bm_score, cosim_score, pagerank_score, pageview_score)
#     ws = np.array([1, 4, 3, 3, 2])
#     if len(query) < 3:
#         ws = np.array([3, 3, 2, 3, 2])
#     # return norm_scores
#     top_d = sorted((combined_scores.keys()), key=lambda x: np.dot(combined_scores[x], ws), reverse=True)[:3]
#     res = [(doc_id, id2title.loc[doc_id]) for doc_id in top_d if doc_id in id2title.index]
#
#     # END SOLUTION
#     return jsonify(res)
#
# @app.route("/search0")
# def search0():
#     """ Returns up to a 100 of your best search results for the query. This is
#         the place to put forward your best search engine, and you are free to
#         implement the retrieval whoever you'd like within the bound of the
#         project requirements (efficiency, quality, etc.). That means it is up to
#         you to decide on whether to use stemming, remove stopwords, use
#         PageRank, query expansion, etc.
#
#         To issue a query navigate to a URL like:
#          http://YOUR_SERVER_DOMAIN/search?query=hello+world
#         where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
#         if you're using ngrok on Colab or your external IP on GCP.
#     Returns:
#     --------
#         list of up to 100 search results, ordered from best to worst where each
#         element is a tuple (wiki_id, title).
#     """
#     res = []
#     query = request.args.get('query', '')
#     if len(query) == 0:
#         return jsonify(res)
#     # BEGIN SOLUTION
#     query = tokenize(query)
#     body_cands, body_cands_dict = get_cands(query, body_idx, "body_idx")
#     title_cands, title_cands_dict = get_cands(query, title_idx, "title_idx")
#     title_cands = title_cands.intersection(body_cands)
#
#     # filter out the candidates that do not have the query terms in the title
#     title_cands_updated = {}
#     for term, pls in title_cands_dict.items():
#         for doc_id in pls.keys():
#             if doc_id in body_cands:
#                 title_cands_updated[term] = pls
#
#     title_cands_dict = title_cands_updated
#
#     title_sim_dict = {doc_id: get_matches(query, title_cands_dict, doc_id) for doc_id in title_cands}
#     title_matches = get_top_n(title_sim_dict)
#
#     bm_score = bm25.search(query, body_cands, body_cands_dict)
#
#     cosim_dict = {doc_id: tf_idf(doc_id, query, body_idx, body_cands_dict) for doc_id in body_cands}
#     cosim_score = get_top_n(cosim_dict)
#
#     relevant_union = set().union(*[body_cands, title_cands])
#
#     pagerank_score = get_top_n(dict([(doc_id, page_rank.get(doc_id, 0)) for doc_id in relevant_union]))
#     pageview_score = get_top_n(dict([(doc_id, page_views.get(doc_id, 0)) for doc_id in relevant_union]))
#
#     # combine all scores
#     combined_scores = combine_scores(relevant_union, title_matches, bm_score, cosim_score, pagerank_score, pageview_score)
#     ws = np.array([1, 4, 3, 3, 2])
#     if len(query) < 3:
#         ws = np.array([3, 3, 2, 3, 2])
#     # return norm_scores
#     top_d = sorted((combined_scores.keys()), key=lambda x: np.dot(combined_scores[x], ws), reverse=True)[:100]
#     res = [(doc_id, id2title.loc[doc_id]) for doc_id in top_d if doc_id in id2title.index]
#
#     # END SOLUTION
#     return jsonify(res)
