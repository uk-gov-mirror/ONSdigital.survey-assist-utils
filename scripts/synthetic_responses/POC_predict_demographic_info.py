#!/usr/bin/env python
import json
from time import gmtime, strftime
from survey_assist_utils.synthetic_responses import (
    instantiate_llm,
    predict_demographic_info,
    construct_demographic_prediction_prompt
)

body = {
    "llm": "gemini",
    "type": "sic",
    "job_title": "baker",
    "job_description": "bake bread and pastries, run the shop",
    "org_description": "a national bakery chain",
}

llm = instantiate_llm()

prompt_demographics = construct_demographic_prediction_prompt(body)
predictions_and_reasoning = predict_demographic_info(llm, prompt_demographics)

predictions, reasoning = predictions_and_reasoning.split("===REASONING===")

print("\nCATEGORY: Prediction | Confidence (0-100):\n------------------------------------------\n", 
      predictions.strip())

categories = ["Age", "Gender", "Disability", "Race / Ethnicity", 
              "Religion or belief", "Sexual Orientation", 
              "Marital Status", "Pregnancy Status", "Gender Alignment"]

demographic_info = {"input": body,
                    "demography": dict(),
                    "reasoning": reasoning.strip()}
for line in predictions.split('\n'):
    try:
        category, response = line.split(":")
        prediction, confidence = response.split("|")
        demographic_info["demography"][category.strip()] = {"prediction": prediction.strip(),
                                                            "confidence": confidence.strip()}
    except ValueError as e:
        pass

with open(f'./demographic_guess_{strftime("%Y-%m-%d|%H.%M.%S", gmtime())}.json', 'w+') as f:
    json.dump(demographic_info, f)