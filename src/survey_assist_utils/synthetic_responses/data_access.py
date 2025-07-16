
import requests
from typing import Tuple, Optional
import json
import os
from ..api_token.jwt_utils import (REFRESH_THRESHOLD,
                                   TOKEN_EXPIRY,
                                   check_and_refresh_token)

def _validate_body(body: dict | str) -> dict:
    if type(body) not in (dict, str):
        raise TypeError(
            "'body' argument must be either a dictionary or a (string) path to a JSON file"
        )

    if type(body) is str:
        body = json.load(body)

    for term in ["llm", "type", "job_title", "job_description", "org_description"]:
        if term not in body:
            raise AttributeError(
                f"key '{term}' is missing from the supplied 'body' argument"
            )
    return body

def get_SA_followup_local(body: dict | str, 
                          url: str="http://0.0.0.0:8080/v1/survey-assist/classify"
                          ) -> Tuple[str, str]:
    """
    Send a survey response to the local API's /classify endpoint to obtain
    the followup question.

    Args:
        body (dict | str): The request body, either as a dictionary or a path to a JSON file.
        url (str, optional): The URL of the /classify endpoint. Defaults to "http://0.0.0.0:8080/v1/survey-assist/classify".

    Returns:
        Tuple[str, str]: A tuple containing the followup question and the reasoning.
    """
    body = _validate_body(body)
    response = requests.post(url, json=body, timeout=20)

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise err

    response_content = json.loads(response.content.decode("utf-8"))
    return (response_content["followup"], response_content["reasoning"])

def get_SA_followup_live(body: dict | str, 
                             current_token: str | None=None,
                             token_start_time: int=-1,
                          ) -> Tuple[str, str, dict]:
    """
    Send a survey response to the live API's /classify endpoint to obtain
    the followup question. 

    Args:
        body (dict | str): The request body, either as a dictionary or a path to a JSON file.
        current_token (str | None, optional): The current JWT token. Defaults to None.
        token_start_time (int, optional): The token's start time. Defaults to -1.

    Returns:
        Tuple[str, str, dict]: A tuple containing the followup question, reasoning, and a dictionary with the current token and its start time.
    """
    body = _validate_body(body)

    api_gateway = os.getenv("API_GATEWAY")
    sa_email = os.getenv("SA_EMAIL")
    jwt_secret_path = os.getenv("JWT_SECRET")
    if (current_token is not None) and (token_start_time != -1):
        token_start_time, current_token = check_and_refresh_token(token_start_time,
                                                                  current_token,
                                                                  jwt_secret_path,
                                                                  api_gateway,
                                                                  sa_email)
    else:
        token_start_time, current_token = check_and_refresh_token(False,
                                                                  "",
                                                                  jwt_secret_path,
                                                                  api_gateway,
                                                                  sa_email)                                           
                                                                  
    url = api_gateway + "/survey-assist/classify"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {current_token}",
    }
    response = requests.post(url, json=body, headers=headers,timeout=20)

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise err

    response_content = json.loads(response.content.decode("utf-8"))
    return (response_content["followup"], 
            response_content["reasoning"], 
            {"current_token": current_token, 
             "token_start_time": token_start_time})

def get_SA_followup_fake(body: dict | str) -> Tuple[str, str, dict]:
    """
    Uses a pre-generated /classify response to act as a mock follow-up
    question and reasoning. 

    Args:
        body (dict | str): The request body, either as a dictionary or a path to a JSON file.

    Returns:
        Tuple[str, str]: A tuple containing the followup question, reasoning.
    """
    body = _validate_body(body)

    fake_response = {'classified': False, 
                     'followup': 'Does your bakery primarily sell its products directly to consumers or to other businesses (e.g., cafes, restaurants)?', 
                     'sic_code': None, 
                     'sic_description': None, 
                     'sic_candidates': [], 
                     'reasoning': "The respondent's description indicates a bakery that bakes bread, cakes, and pastries.  This aligns strongly with SIC code 10710, 'Manufacture of bread; manufacture of fresh pastry goods and cakes'.  However, if the bakery's primary activity is retail sales rather than manufacturing, SIC code 47240 would be more appropriate.  The follow-up question aims to clarify this distinction.", 
                     'prompt_used': "{'industry_descr': 'small scale, independently owned bakery', 'job_title': 'baker', 'job_description': 'Bake bread, cakes, pastries', 'sic_index': '{Code: 46690, Title: Wholesale of other machinery and equipment, Example activities: Bakery ovens (wholesale)}\\n{Code: 10710, Title: Manufacture of bread; manufacture of fresh pastry goods and cakes, Example activities: Bakery (baking main activity) (manufacture), Manufacture of bread; manufacture of fresh pastry goods and cakes, Cakes (manufacture)}\\n{Code: 46360, Title: Wholesale of sugar and chocolate and sugar confectionery, Example activities: Bakery products (wholesale), Bread (wholesale), Flour confectionery (wholesale)}\\n{Code: 28930, Title: Manufacture of machinery for food, beverage and tobacco processing, Example activities: Bakery machinery and ovens (manufacture), Bakery ovens (industrial) (manufacture), Bakery moulders (manufacture)}\\n{Code: 47240, Title: Retail sale of bread, cakes, flour confectionery and sugar confectionery in specialised stores, Example activities: Bakery (selling main activity) (retail), Baker (retail), Cakes (retail)}'}"}
    return fake_response['followup'], fake_response['reasoning']