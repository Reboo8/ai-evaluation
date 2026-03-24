from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import START, END
from dotenv import load_dotenv
from typing import List
import os
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

class state(TypedDict):
    resume_text: str
    assessment_text: NotRequired[str]
    interview_text: str
    resume_summary: NotRequired[str]
    assessment_summary: NotRequired[str]
    interview_summary: NotRequired[str]

    resume_score: NotRequired[float]
    assessment_score: NotRequired[float]
    interview_score: NotRequired[float]

    resume_strengths: NotRequired[list[str]]
    resume_weaknesses: NotRequired[list[str]]

    assessment_strengths: NotRequired[list[str]]
    assessment_weaknesses: NotRequired[list[str]]

    interview_strengths: NotRequired[list[str]]
    interview_weaknesses: NotRequired[list[str]]

    strengths: NotRequired[list[str]]
    weaknesses: NotRequired[list[str]]

    overall_summary: NotRequired[str]
    final_score: NotRequired[float]

class OutputSchema(BaseModel):
    summary: str
    score: float
    strengths: List[str]
    weaknesses: List[str]

class FinalSchema(BaseModel):
    summary: str
    score: float

class EvaluationRequest(BaseModel):
    resume_text: str
    assessment_text: str
    interview_text: str


class EvaluationResponse(BaseModel):
    resume_summary: str
    resume_score: float
    assessment_summary: str
    assessment_score: float
    interview_summary: str
    interview_score: float
    overall_summary: str
    final_score: float
    strengths: List[str]
    weaknesses: List[str]


app = FastAPI(title="self assessment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=groq_api_key)
structured_llm = llm.with_structured_output(OutputSchema)
final_llm = llm.with_structured_output(FinalSchema)

resume_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that analyzes resumes scores and tips and provides a summary, a score out of 10, and lists of strengths and weaknesses."),
        ("human", "{resume}")
    ]
)

assessment_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that analyzes assessment results scores and tips and provides a summary and a score out of 10, and list of all the strengths and weaknesses."),
        ("human", "{assessment}")
    ]
)

interview_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that analyzes interview transcripts scores and tips and provides a summary, a score out of 10, and lists of strengths and weaknesses."),
        ("human", "{interview}")
    ]
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that takes the resume and interview analysis of a candidate and provides the final overall summary and score for the candidate."),
        ("human", "Resume Analysis: {resume_summary}, Resume Score: {resume_score}, Interview Analysis: {interview_summary}, Interview Score: {interview_score}, Strengths: {strengths}, Weaknesses: {weaknesses}")
    ]
)

graph_builder = StateGraph(state)

async def resume_llm(state: state) -> state:

    chain = resume_prompt | structured_llm
    result = await chain.ainvoke({"resume": state["resume_text"]})
    return {
        "resume_summary": result.summary,
        "resume_score": result.score,
        "resume_strengths": result.strengths,
        "resume_weaknesses": result.weaknesses
    }

async def assessment_llm(state: state) -> state:
    chain = assessment_prompt | structured_llm
    result = await chain.ainvoke({"assessment" : state["assessment_text"]})
    return {
        "assessment_summary": result.summary,
        "assessment_score": result.score,
        "assessment_strengths": result.strengths,
        "assessment_weaknesses": result.weaknesses
    }

async def interview_llm(state: state) -> state:

    chain = interview_prompt | structured_llm
    result = await chain.ainvoke({"interview": state["interview_text"]})

    return {
        "interview_summary": result.summary,
        "interview_score": result.score,
        "interview_strengths": result.strengths,
        "interview_weaknesses": result.weaknesses
    }

async def final_node(state: state) -> state:

    strengths = (
        state.get("resume_strengths", []) +
        state.get("interview_strengths", []) +
        state.get("assessment_strengths", [])
    )

    weaknesses = (
        state.get("resume_weaknesses", []) +
        state.get("interview_weaknesses", []) +
        state.get("assessment_weaknesses", [])
    )

    chain = final_prompt | final_llm
    result = await chain.ainvoke({
        "resume_summary": state["resume_summary"],
        "resume_score": state["resume_score"],
        "assessment_summary": state["assessment_summary"],
        "assessment_score": state["assessment_score"],
        "interview_summary": state["interview_summary"],
        "interview_score": state["interview_score"],
        "strengths": strengths,
        "weaknesses": weaknesses
    })
    return {
        "overall_summary": result.summary,
        "final_score": result.score,
        "strengths": strengths,
        "weaknesses": weaknesses
    }

graph_builder.add_node("resume_node", resume_llm)
graph_builder.add_node("assessment_node", assessment_llm)
graph_builder.add_node("interview_node", interview_llm)
graph_builder.add_node("final_node", final_node)

graph_builder.add_edge(START, "resume_node")
graph_builder.add_edge(START, "assessment_node")
graph_builder.add_edge(START, "interview_node")
graph_builder.add_edge("resume_node", "final_node")
graph_builder.add_edge("assessment_node", "final_node")
graph_builder.add_edge("interview_node", "final_node")
graph_builder.add_edge("final_node", END)

graph = graph_builder.compile()

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_candidate(data: EvaluationRequest):
    try:
        input_data = {
            "resume_text": data.resume_text,
            "assessment_text": data.assessment_text,
            "interview_text": data.interview_text
        }

        result = await asyncio.wait_for(
            graph.ainvoke(input_data),
            timeout=120
        )

        return {
            "resume_summary": result.get("resume_summary"),
            "resume_score": result.get("resume_score"),
            "assessment_summary": result.get("assessment_summary"),
            "assessment_score": result.get("assessment_score"),
            "interview_summary": result.get("interview_summary"),
            "interview_score": result.get("interview_score"),
            "overall_summary": result.get("overall_summary"),
            "final_score": result.get("final_score"),
            "strengths": result.get("strengths", []),
            "weaknesses": result.get("weaknesses", []),
        }

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="LLM request timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))