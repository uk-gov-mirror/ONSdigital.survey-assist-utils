def _persona_prompt(persona):
    """Constructs a section of the LLM prompt, informing it about
    the personal characteristics (if any) of the persona it should
    emulate.
    """
    if persona is None:
        return """
        You are a UK worker, responding to a survey surrounding industrial
        classification for the use in the UK official statistics.
        You always respond in British English."""

    else:
        # TODO
        pass


def _reminder_prompt(request_body: dict, followup_question: str):
    """Constructs a section of the LLM prompt which 'reminds' it about the job
    held by the person it is pretending to be, and asks it to respond to the
    provided follow-up question.
    """
    return f"""
    Below is a reminder of the main activity your
    company does, your job title and job description.
    The survey interviewer has not been able to fully classify your company's
    Standard Industrial Classification (SIC) from the Job Data you have provided so far,
    and so they have asked you a clarifying question.
    Please answer this clarifying question.

    ===Your Job Data===
    - Company's main activity: {request_body['org_description']}
    - Job Title: {request_body['job_title']}
    - Job Description: {request_body['job_description']}

    ===Survey Data===
    - clarifying question: {followup_question}
    """
