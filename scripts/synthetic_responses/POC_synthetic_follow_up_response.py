#!/usr/bin/env python

import requests
import json
from langchain_google_vertexai import VertexAI
from langchain.chains.llm import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.prompt import PromptTemplate
from survey_assist_utils.synthetic_responses import (_persona_prompt, 
                                                     _reminder_prompt,
                                                     instantiate_llm,
                                                     get_followup,
                                                     construct_prompt,
                                                     answer_followup)

body = {
  "llm": "gemini",
  "type": "sic",
  "job_title": "baker",
  "job_description": "bake bread and pastries, run the shop",
  "org_description": "a national bakery chain"
}

llm = instantiate_llm()
persona = None

followup, has_followup, reasoning = get_followup(body)
PROMPT_FOLLOWUP = construct_prompt(persona, body, followup)
answer = answer_followup(llm, PROMPT_FOLLOWUP)

print("PROMPT:\n",PROMPT_FOLLOWUP)
print("\nANSWER:\n",answer)