#!/usr/bin/env python

import json

import requests
from langchain_google_vertexai import VertexAI
from typing import Tuple, Optional
from .prompts import _persona_prompt, _reminder_prompt

class SyntheticResponder:
    """
    This class provides functionality for generating synthetic responses to survey questions,
    particularly follow-up questions, using a specified persona and a Large Language Model (LLM).

    Attributes:
        persona (optional): A dictionary describing the demographic characteristics of the persona
                           the LLM should emulate. 
                           Defaults to None.
        get_question_function (callable): A function that retrieves the follow-up question from an API.
                                          It should take a parameter 'body' as a first argument, although
                                          it can accept additional arguments.
        model_name (str): The name of the LLM to use. 
                          Defaults to "gemini-1.5-flash".
        llm: An instance of the LLM (currently VertexAI) used for generating responses.

    Methods:
        instantiate_llm:
            Initializes a VertexAI instance. Defaults to "gemini-1.5-flash".
        construct_prompt:
            Constructs the LLM prompt to answer a follow-up question, incorporating the persona
            and survey information.
        answer_followup:
            Gets the LLM's response to the follow-up question, using the provided prompt.
    """

    def __init__(self, 
                 get_question_function: callable,
                 persona: Optional[dict] = None, 
                 model_name="gemini-1.5-flash"):
        self.persona = persona
        self.get_question_function = get_question_function
        self.model_name = model_name
        self.instantiate_llm(model_name=self.model_name)

    def instantiate_llm(self, model_name: str = "gemini-1.5-flash"):
        """Initialises a VertexAI instance."""
        self.llm = VertexAI(
            model_name=model_name,
            max_output_tokens=1_600,
            temperature=0.0,
            location="europe-west2",
        )

    def construct_prompt(self, body: dict | str, followup: str) -> str:
        """Constructs and LLM prompt to respond to the followup question in a specified persona."""
        if type(body) not in (dict, str):
            raise TypeError(
                "'body' argument must be either a dictionary or a (string) path to a JSON file"
            )
        if type(body) is str:
            body = json.load(body)
        if type(followup) is str:
            return _persona_prompt(self.persona) + _reminder_prompt(body, followup)
        else: 
            raise ValueError("No follow-up question provided.")


    def answer_followup(self, prompt: str) -> str:
        """Gets the LLM's response to the followup question,
        as specified in the constructed prompt.
        """
        return self.llm.invoke(prompt)
