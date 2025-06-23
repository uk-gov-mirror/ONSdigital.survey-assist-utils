#!/usr/bin/env python

import requests
import json
from langchain_google_vertexai import VertexAI
from langchain.chains.llm import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.prompt import PromptTemplate
from .prompts import _persona_prompt, _reminder_prompt

def instantiate_llm(model_name: str="gemini-1.5-flash"):
  """Initialises a VertexAI instance"""
  return VertexAI(model_name=model_name,
                  max_output_tokens=1_600,
                  temperature=0.0,
                  location="europe-west2")

def get_followup(body: dict|str, classify_endpoint_url=None):
  """Send a survey response to the API's /classify endpoint to obtain
  the followup question.
  """
  if classify_endpoint_url is None:
    classify_endpoint_url = "http://0.0.0.0:8080/v1/survey-assist/classify"

  if type(body) not in (dict, str):
    raise TypeError("'body' argument must be either a dictionary or a (string) path to a JSON file")

  if type(body) is str:
    body = json.load(body)
  
  for term in ['llm','type','job_title','job_description','org_description']:
    if term not in body:
      raise AttributeError(f"key '{term}' is missing from the supplied 'body' argument")

  response = requests.post(classify_endpoint_url, json=body, verify=False)
  response_content = json.loads(response.content.decode('utf-8'))

  has_followup = False if response_content["classified"] else True
  followup = response_content["followup"]
  reasoning = response_content["reasoning"]

  return (followup, has_followup, reasoning)

def construct_prompt(persona, body, followup):
  """Constructs and LLM prompt to respond to the followup question in a specified persona"""
  return _persona_prompt(persona) + _reminder_prompt(body, followup)

def answer_followup(llm, prompt: str):
  """Gets the LLM's response to the followup question, 
  as specified in the constructed prompt"""
  return llm.invoke(prompt)
