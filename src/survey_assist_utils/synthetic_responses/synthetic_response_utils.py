#!/usr/bin/env python

import json

import requests
from langchain_google_vertexai import VertexAI

from .prompts import _persona_prompt, _predict_characteristics_prompt, _reminder_prompt


def instantiate_llm(model_name: str = "gemini-1.5-flash"):
    """Initialises a VertexAI instance."""
    return VertexAI(
        model_name=model_name,
        max_output_tokens=1_600,
        temperature=0.0,
        location="europe-west2",
    )

def load_request_body(fname: str):
    """Load a JSON file as a request body for querying the /classify endpoint."""
    try:
        with open(fname, 'r') as f:
            body = json.load(f)
    except FileNotFoundError as e:
        print(f"File {fname} was not found")
        raise e
    return body


def get_followup(body: dict, classify_endpoint_url=None):
    """Send a survey response to the API's /classify endpoint to obtain
    the followup question.
    """
    if classify_endpoint_url is None:
        classify_endpoint_url = "http://0.0.0.0:8080/v1/survey-assist/classify"

    if type(body) is not dict:
        raise TypeError(
            "'body' argument must be a dictionary. Use load_request_body() to read in a local JSON file"
        )

    for term in ["llm", "type", "job_title", "job_description", "org_description"]:
        if term not in body:
            raise AttributeError(
                f"key '{term}' is missing from the supplied 'body' argument"
            )

    response = requests.post(classify_endpoint_url, json=body, timeout=10)
    response_content = json.loads(response.content.decode("utf-8"))

    has_followup = not response_content["classified"]
    followup = response_content["followup"]
    reasoning = response_content["reasoning"]

    return (followup, has_followup, reasoning)


def construct_prompt(persona, body, followup):
    """Constructs an LLM prompt to respond to the followup question in a specified persona."""
    return _persona_prompt(persona) + _reminder_prompt(body, followup)


def construct_demographic_prediction_prompt(body):
    """Constructs an LLM prompt to predict the demographic
    characteristics of a survey respondent.
    """
    return _predict_characteristics_prompt(body)


def answer_followup(llm, prompt: str):
    """Gets the LLM's response to the followup question,
    as specified in the constructed prompt.
    """
    return llm.invoke(prompt)


def predict_demographic_info(llm, prompt: str):
    """Gets the LLM's response to the followup question,
    as specified in the constructed prompt.
    """
    return llm.invoke(prompt)
