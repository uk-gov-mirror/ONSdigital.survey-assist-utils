def _persona_prompt(persona):
    """Constructs a section of the LLM prompt, informing it about
    the personal characteristics (if any) of the persona it should
    emulate"""
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
    provided follow-up question"""
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

def _predict_characteristics_prompt(request_body: dict):
    """Constructs an LLM prompt which requests the LLM to
    predict (protected) demographic information about a survey 
    respondent based on their responses."""

    return f"""
    Below (under Job Data) is the survey response from an individual; including
    the main activity of their company, their job title and their job 
    description. 
    There are several demographic characteristics (under Demographic Characteristics)
    about the individual which we do not know; please predict them based on the 
    Job Data, and assign confidence levels to each prediction. Please use the following format:

    <characteristic name>: <prediction (only one of the options in parentheses for the characteristic)> | <confidence (0 meaning no confidence, 100 meaning certain)>

    For the Age characteristic do not predict an age range, only a single integer.
    We understand all potential caveats and nuances of the predictions, do not add additional information 
    to the requested output format; instead append any reasoning to the end of your reply, after a line of the following format:

    ===REASONING===

    ===Job Data===
    - Company's main activity: {request_body['org_description']}
    - Job Title: {request_body['job_title']}
    - Job Description: {request_body['job_description']}

    ===Demographic Characteristics===
    - Age (integer)
    - Gender ('M' for male, 'F' for female or 'NB' for non-binary)
    - Disability ('Y' for yes, 'N' for no)
    - Race / Ethnicity 
        ("Asian, Asian British, Asian Welsh" or
        "Black, Black British, Black Welsh, Caribbean or African" or
        "Mixed or Multiple" or
        "White" or
        "Other ethnic group")
    - Religion or belief (Buddhist, Christian, 
                          Hindu, Jewish, 
                          Muslim, Sikh
                          Other religion, No religion)
    - Sexual Orientation (Straight / Heterosexual,
                          Gay / Lesbian / Homosexual,
                          Bisexual,
                          Asexual,
                          Other)
    - Marital Status (Single, Married,
                      Divorced, Widow / Widower)
    - Pregnancy Status (Pregnant, Not Pregnant)
    - Gender Alignment (Transgender, Cisgender, Other)
    """

