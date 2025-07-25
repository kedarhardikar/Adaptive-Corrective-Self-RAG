# $env:PYTHONPATH="."
# >> poetry run pytest graph/chains/tests/test_chains.py
from dotenv import load_dotenv

load_dotenv()


from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import (GradeHallucination,
                                               hallucination_grader)
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.router import RouteQuery, question_router
from ingestion import retriever
from pprint import pprint


# def test_retrival_grader_answer_yes() -> None:
#     question = "agent memory"
#     docs = retriever.invoke(question)
#     doc_txt = docs[1].page_content

#     res: GradeDocuments = retrieval_grader.invoke(
#         {"question": question, "document": doc_txt}
#     )
#     print("Final answer",res.binary_score)

#     assert res.binary_score.lower() == "yes"

    
# def test_retrival_grader_answer_no() -> None:
#     question = "How to drive a car"
#     docs = retriever.invoke(question)
#     doc_txt = docs[1].page_content

#     res: GradeDocuments = retrieval_grader.invoke(
#         {"question": question, "document": doc_txt}
#     )
#     print("Final answer",res.binary_score)

#     assert res.binary_score.lower() == "no"


# def test_generation_chain() -> None:
#     question = "agent memory"
#     docs = retriever.invoke(question)
#     generation = generation_chain.invoke({"context": docs, "question": question})
#     pprint(generation)


# def test_hallucination_grader_answer_yes() -> None:
#     question = "agent memory"
#     docs = retriever.invoke(question)

#     generation = generation_chain.invoke({"context": docs, "question": question})
#     res: GradeHallucination = hallucination_grader.invoke(
#         {"documents": docs, "generation": generation}
#     )
#     assert res.binary_score


# def test_hallucination_grader_answer_no() -> None:
#     question = "agent memory"
#     docs = retriever.invoke(question)

#     res: GradeHallucination = hallucination_grader.invoke(
#         {
#             "documents": docs,
#             "generation": "In order to make pizza we need to first start with the dough",
#         }
#     )
#     assert not res.binary_score


def test_router_to_vectorstore() -> None:
    question = "agent memory"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "how to make pizza"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"