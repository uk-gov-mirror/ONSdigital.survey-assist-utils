
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from survey_assist_utils import get_logger
from survey_assist_utils.api_token.jwt_utils import (
    REFRESH_THRESHOLD,
    TOKEN_EXPIRY,
)
from survey_assist_utils.synthetic_responses.synthetic_response_utils import SyntheticResponder
from survey_assist_utils.synthetic_responses.data_access import (
    get_SA_followup_local,
    get_SA_followup_live, 
    get_SA_followup_fake,
)

logger = get_logger(__name__)

@pytest.mark.synthetic_responses
def test_create_synthetic_responder_object():
    """Test creating a SyntheticResponder object."""
    arbitrary_func = lambda *args: None
    responder = SyntheticResponder(arbitrary_func, persona=None, model_name="gemini-1.5-flash")
    assert responder.model_name == "gemini-1.5-flash"
    assert responder.llm is not None
    assert responder.persona is None

@pytest.mark.synthetic_responses
def test_receive_question():
    """Test SyntheticResponder can get a question."""
    responder = SyntheticResponder(get_SA_followup_fake, persona=None, model_name="gemini-1.5-flash")
    body = {
        "llm": "gemini",
        "type": "sic",
        "job_title": "baker",
        "job_description": "Bake bread, cakes, pastries",
        "org_description": "small scale, independently owned bakery"
    }
    question, reasoning = responder.get_question_function(body)
    assert question is not None
    assert reasoning is not None
    assert question == 'Does your bakery primarily sell its products directly to consumers or to other businesses (e.g., cafes, restaurants)?'

@pytest.mark.synthetic_responses
def test_construct_prompt():
    """Test SyntheticResponder can construct a prompt."""
    responder = SyntheticResponder(get_SA_followup_fake, persona=None, model_name="gemini-1.5-flash")
    body = { 
        "llm": "gemini",
        "type": "sic",
        "job_title": "baker",
        "job_description": "Bake bread, cakes, pastries",
        "org_description": "small scale, independently owned bakery"
    }
    question, reasoning = responder.get_question_function(body)
    prompt = responder.construct_prompt(body, question)
    assert prompt is not None
    assert prompt.startswith("""
        You are a UK worker, responding to a survey surrounding industrial
        classification for the use in the UK official statistics.
        You always respond in British English.""")
    assert """
    Below is a reminder of the main activity your
    company does, your job title and job description.
    The survey interviewer has not been able to fully classify your company's
    Standard Industrial Classification (SIC) from the Job Data you have provided so far,
    and so they have asked you a clarifying question.
    Please answer this clarifying question.""" in prompt
    assert question in prompt
    assert body["job_title"] in prompt
    assert body["job_description"] in prompt
    assert body["org_description"] in prompt

@pytest.mark.synthetic_responses
def test_answer_followup():
    """Test SyntheticResponder can answer a followup question."""
    responder = SyntheticResponder(get_SA_followup_fake, persona=None, model_name="gemini-1.5-flash")
    body = {
        "llm": "gemini",
        "type": "sic",
        "job_title": "baker",
        "job_description": "Bake bread, cakes, pastries",
        "org_description": "small scale, independently owned bakery"
    }
    question, reasoning = responder.get_question_function(body)
    prompt = responder.construct_prompt(body, question)
    answer = responder.answer_followup(prompt)
    assert answer is not None

@pytest.skip(reason="requires local vector store and API to be running", allow_module_level=True)
@pytest.mark.synthetic_responses
def test_local_api():
    """Test SyntheticResponder can use locally running SA API."""
    responder = SyntheticResponder(get_SA_followup_local, persona=None, model_name="gemini-1.5-flash")
    body = {
        "llm": "gemini",
        "type": "sic",
        "job_title": "baker",
        "job_description": "Bake bread, cakes, pastries",
        "org_description": "small scale, independently owned bakery"
    }
    question, reasoning = responder.get_question_function(body)
    prompt = responder.construct_prompt(body, question)
    answer = responder.answer_followup(prompt)
    assert answer is not None

@pytest.skip(reason="requires environmental variables to be set", allow_module_level=True)
@pytest.mark.synthetic_responses
def test_live_api():
    """Test SyntheticResponder can use externally running PoC SA API."""
    responder = SyntheticResponder(get_SA_followup_live, persona=None, model_name="gemini-1.5-flash")
    body = {
        "llm": "gemini",
        "type": "sic",
        "job_title": "baker",
        "job_description": "Bake bread, cakes, pastries",
        "org_description": "small scale, independently owned bakery"
    }
    question, reasoning, token_info = responder.get_question_function(body)
    prompt = responder.construct_prompt(body, question)
    answer = responder.answer_followup(prompt)
    assert answer is not None

