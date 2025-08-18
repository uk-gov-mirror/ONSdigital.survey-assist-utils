import marimo

__generated_with = "0.14.16"
app = marimo.App(
    width="medium",
    layout_file="layouts/sic_review_marimo.grid.json",
)


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import json
    import ast
    import os
    from datetime import datetime
    # from collections import namedtuple
    # import re
    return datetime, mo, os, pd


@app.cell
def _(mo):
    mo.md("""# 📊 SIC Code Review Tool""")
    return


@app.cell
def _():
    # def parse_sic_candidates(candidates_str):
    #     if pd.isna(candidates_str) or candidates_str == "" or str(candidates_str).lower() == "nan":
    #         return []

    #     try:
    #         # Extract all RagCandidate entries using regex
    #         pattern = r"([0-9]+x*X*)"
    #         matches = re.findall(pattern, str(candidates_str))

    #         return matches
    #     except Exception as e:
    #         raise
    return


@app.cell
def _(mo):
    # File path - EDIT THIS TO YOUR ACTUAL FILE PATH
    #CSV_FILE_PATH = "D:/survey-assist-utils/data/intermediate_results_rand100.csv"
    CSV_FILE_PATH = "./two_prompt_outputs/STG5.parquet"


    mo.md(f"""
    ## File Configuration
    **Current file path:** `{CSV_FILE_PATH}`

    *Edit the CSV_FILE_PATH variable in the cell above to point to your data file.*
    """)
    return (CSV_FILE_PATH,)


@app.cell(hide_code=True)
def _(CSV_FILE_PATH, mo, os, pd):
    initial_data_df = None
    _load_response = None
    try:
        if os.path.exists(CSV_FILE_PATH):
            initial_data_df = pd.read_parquet(CSV_FILE_PATH)

            # Add review columns if missing
            review_columns = ['reviewer_initials', 'review_timestamp', 'model_prediction_plausible', 
                             'reviewer_notes']

            for col in review_columns:
                if col not in initial_data_df.columns:
                    if col in ['model_prediction_plausible']:
                        initial_data_df[col] = None
                    else:
                        initial_data_df[col] = ""

            entry_count = len(initial_data_df)
            review_count = sum(initial_data_df['reviewer_initials'].notna() & (initial_data_df['reviewer_initials'] != ""))

            _load_response = mo.md(f"""
            ## Data Loaded Successfully!

            - **Total entries:** {entry_count}
            - **Already reviewed:** {review_count}  
            """)
        else:
            _load_response = mo.md(f"**File not found at:** `{CSV_FILE_PATH}`\n\nPlease check the file path and update the CSV_FILE_PATH variable.")

    except Exception as e:
        _load_response = mo.md(f"**Error loading file:** {str(e)}")
    _load_response
    return (initial_data_df,)


@app.cell
def _(initial_data_df):
    # Filter out entries where CC provided a single 5-digit SIC code
    data_df = initial_data_df[initial_data_df["Unambiguous"] == False]
    return (data_df,)


@app.cell
def _(data_df, mo):
    # Navigation slider
    entry_slider = None
    _slider_response = None
    if data_df is not None:
        entry_slider = mo.ui.slider(
            start=1,
            stop=len(data_df),
            value=1,
            label=f"Select Entry (1 to {len(data_df)}):",
            show_value=True
        )

        _slider_response = mo.md(f"""
        ## Navigation

        Use the slider below to navigate between entries:

        {entry_slider}
        """)
    else:
        _slider_response = mo.md("**No navigation available** - Please load data first")
    _slider_response

    return (entry_slider,)


@app.cell
def _(data_df, entry_slider, mo, pd):
    # Display current entry
    _entry_response = None
    if data_df is not None and entry_slider is not None:
        current_idx = entry_slider.value - 1
        current_row = data_df.iloc[current_idx]

        # Basic information
        job_title = current_row.get('soc2020_job_title', 'N/A')
        industry = current_row.get('sic2007_employee', 'N/A')
        model_sic = current_row.get('final_sic', 'N/A')
        model_sic_higher = current_row.get('higher_level_final_sic', 'N/A')
        job_desc = str(current_row.get('soc2020_job_description', 'N/A'))
        followup_q = current_row.get('followup_question', 'N/A')
        followup_a = current_row.get('followup_answer', 'N/A')

        # Parse SIC candidates
        candidates_display = "No candidates available"
        try:
            candidates_raw = current_row.get('alt_sic_candidates', '')
            #candidates_list = [cr.code for cr in candidates_raw] #parse_sic_candidates(candidates_raw)
        
            # Format candidates for display
            if len(candidates_raw) > 0:
                candidates_display = ""
                for i, candidate in enumerate(candidates_raw, 1):
                    candidates_display += f"**Candidate {i}:** `{candidate['code']} | {candidate['title']}`\n\n"
                    # candidates_display += f"**{candidate.class_code}** - {candidate.class_descriptive}\n"
                    # candidates_display += f"   *Likelihood: {candidate.likelihood:.1%}*\n\n"
        except Exception as e:
            candidates_display = f"Error parsing candidates: {str(e)}"

        # Review status
        has_review = pd.notna(current_row.get('reviewer_initials')) and current_row.get('reviewer_initials') != ""
        status_text = "REVIEWED" if has_review else "NOT REVIEWED"
        header_section = mo.md(f"""## Entry {current_idx + 1} of {len(data_df)} - {status_text}
        """)

        original_response_section = mo.md(f"""### Original Data

        **Job Title:** {job_title}  
        **Industry:** {industry}  
        **Job Description:**
        {job_desc[:500]}{"..." if len(job_desc) > 500 else ""}
        ----------
        """)

        follow_up_section = mo.md(f"""### Follow-up information

        **Question:** {followup_q}  
        **Answer:** {followup_a}
        ----------
        """)

        candidate_section = mo.md(f"""### SIC Candidates

        {candidates_display}
        ----------
        """)

        model_prediction = mo.md(f"""### Model Prediction

        **Model SIC Code:** `{model_sic}{model_sic_higher}`
        """)


        # review_section = mo.md(f"""### Existing Review:

        # **Reviewer:** {current_row.get('reviewer_initials', 'None')}  
        # **Model Plausible:** {current_row.get('model_prediction_plausible', 'Not answered')}  
        # **Better Available:** {current_row.get('better_sic_available', 'Not answered')}  
        # **Recommended SIC:** {current_row.get('recommended_sic_code', 'None')}  
        # **Notes:** {current_row.get('reviewer_notes', 'None')}
        # """)

        _entry_response = mo.vstack(
            [
                header_section,
                original_response_section,
                follow_up_section,
                candidate_section,
                model_prediction,
                # review_section
            ]

        )
    else:
        _entry_response = mo.md("⚠️ **No entry to display** - Please check that data is loaded and an entry is selected")

    _entry_response

    return


@app.cell
def _(mo):
    # Review input form
    form = (
        mo.md('''## 📝 Review Form"

        {reviewer_initials}

        {model_plausible}

        {review_notes}
    ''')
        .batch(
            reviewer_initials=mo.ui.text(label="Your initials"),
            model_plausible = mo.ui.radio(
                            options=["Yes", "No"],
                            label="Is the model prediction plausible?",

            ),
            # better_available = mo.ui.radio(
            #                 options=["Yes", "No"],
            #                 label="Is a better SIC code available?"
            #             ),

            # recommended_sic = mo.ui.text(
            #                 label="Recommended SIC Code (if applicable):",
            #                 placeholder="Enter SIC code if you have a better suggestion"
            #             ),

            review_notes = mo.ui.text_area(
                            label="Notes:",
                            placeholder="Any additional comments or observations"
                        ),
        )
        .form(show_clear_button=True, clear_on_submit=True, bordered=False)
    )

    form
    return (form,)


@app.cell
def _(CSV_FILE_PATH, data_df, datetime, entry_slider, form, mo):
    # Handle review submission
    _warning = None
    if form.value and data_df is not None and entry_slider is not None:
        # Validation
        if not form.value['reviewer_initials'] or not form.value['reviewer_initials'].strip():
            _warning = mo.md("❌ **Please enter your initials**")
        elif not form.value['model_plausible']:
            _warning = mo.md("❌ **Please answer whether the model prediction is plausible**")
        # elif not form.value['better_available']:
        #     _warning = mo.md("❌ **Please answer whether a better SIC code is available**")
        else:
            # Save the review
            review_idx = entry_slider.value - 1
            data_df.loc[review_idx, 'reviewer_initials'] = form.value['reviewer_initials'].strip()
            data_df.loc[review_idx, 'review_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data_df.loc[review_idx, 'model_prediction_plausible'] = (form.value['model_plausible'] == "Yes")
            # data_df.loc[review_idx, 'better_sic_available'] = (form.value['better_available'] == "Yes")
            # data_df.loc[review_idx, 'recommended_sic_code'] = form.value['recommended_sic'].strip() if form.value['recommended_sic'] else ""
            data_df.loc[review_idx, 'review_notes'] = form.value['review_notes'].strip() if form.value['review_notes'] else ""
            data_df.to_csv(CSV_FILE_PATH, index=False)
            _warning = mo.md(f"✅ **Review saved successfully for entry {entry_slider.value}!**")
    _warning

    return


@app.cell
def _(data_df, mo):
    # Progress summary
    _progress = None
    if data_df is not None:
        total_entries = len(data_df)
        completed = sum(data_df['reviewer_initials'].notna() & (data_df['reviewer_initials'] != ""))
        progress_percent = (completed / total_entries * 100) if total_entries > 0 else 0

        # Text-based progress bar
        filled_blocks = int(progress_percent / 5)
        progress_bar = "█" * filled_blocks + "░" * (20 - filled_blocks)

        _progress = mo.md(f"""
        ## 📊 Review Progress

        **Completed:** {completed} / {total_entries} entries ({progress_percent:.1f}%)

        `{progress_bar}` {progress_percent:.1f}%

        ### Quick Stats:
        - **Remaining:** {total_entries - completed} entries
        - **Average per day:** Complete ~50 entries to finish in 2-3 weeks
        """)
    else:
        _progress = mo.md("")

    _progress

    return


if __name__ == "__main__":
    app.run()
