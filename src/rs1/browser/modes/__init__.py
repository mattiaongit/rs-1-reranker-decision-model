"""Decision modes, from the reranker alone to the reranker plus generic rules:

    pure       only the help jev-ultrafast gives Jev
    assist     code prepares the options, the reranker always chooses
    pure_done  the reranker chooses every move, plus a done check
    rules      the reranker ranks every action, generic rules can override it
"""
