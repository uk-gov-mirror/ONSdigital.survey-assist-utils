#!/usr/bin/env python
from survey_assist_utils.synthetic_responses.synthetic_response_utils import (
    answer_followup,
    construct_prompt,
    get_followup,
    instantiate_llm,
)

body = {
    "llm": "gemini",
    "type": "sic",
    "job_title": "baker",
    "job_description": "bake bread and pastries, run the shop",
    "org_description": "a national bakery chain",
}

llm = instantiate_llm()
persona = None

followup, has_followup, reasoning = get_followup(
    body
)  # todo: add notes on URL, endpoint & prompt
PROMPT_FOLLOWUP = construct_prompt(persona, body, followup)
answer = answer_followup(llm, PROMPT_FOLLOWUP)

print("PROMPT:\n", PROMPT_FOLLOWUP)
print("\nANSWER:\n", answer)
