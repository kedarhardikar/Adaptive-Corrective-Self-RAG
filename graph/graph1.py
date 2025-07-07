# 🔥 FIRST: load the .env BEFORE anything else
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langgraph.graph import END, StateGraph
from graph.chains.router import question_router, RouteQuery


from graph.nodes import generate, grader, retrieve, web_search

from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader

from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.state import GraphState

def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE


def decide_to_generate(state):
    """ DECIDES WHETHER TO PASS TO GENERATE NODE OR WEBSEARCH NODE"""

    print("ASSCESS GRADED DOCS")

    if state['web_search']:
        print("NOT ALL DOCS RELEVANT TO THE QUESTION->WEBSEARCH")
        return WEBSEARCH  #return "websearch"
    
    else:
        print("ALL DOCUMENTS RELEVANT TO THE QUESTION")
        return GENERATE
    
def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:  #if hallucination_grade == True
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"    #No hallucination, right answer
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"     #No hallucinations but wrong answer
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"   #Hallucinations


print("Inside the graph")    
workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grader)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)


workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)
workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE,GRADE_DOCUMENTS)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH, WEBSEARCH,
        GENERATE, GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,    #Hallucinations
        "useful": END,     #No hallucination, right answer
        "not useful": WEBSEARCH,    #No hallucinations but wrong answer
    },
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()