from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState

def grader(GraphState):
    """
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """
    
    
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

    docs = GraphState['documents']
    question = GraphState['question']

    WEB_SEARCH = False
    filtered_docs = []

    for d in docs:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
            )
        grade = score.binary_score

        if grade.lower() == "yes":
            filtered_docs.append(d)
        else:
            WEB_SEARCH = True
            continue

    return {"documents": filtered_docs, "question": question, "web_search": WEB_SEARCH}

