import marimo

__generated_with = "0.9.34"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import json
    import ast
    import os
    from datetime import datetime
    return ast, datetime, json, mo, os, pd


@app.cell
def __():
    mo.md("# 📊 SIC Code Review Tool")
    return


@app.cell
def __():
    # File path - EDIT THIS TO YOUR ACTUAL FILE PATH
    CSV_FILE_PATH = "/home/user/survey-assist-utils/data/evaluation_data/dummy_eval_data.csv"
    
    mo.md(f"""
    ## 📁 File Configuration
    **Current file path:** `{CSV_FILE_PATH}`
    
    *Edit the CSV_FILE_PATH variable in the cell above to point to your data file.*
    """)
    return CSV_FILE_PATH,


@app.cell
def __(CSV_FILE_PATH, pd, os, mo):
    # Load data with detailed feedback
    data_df = None
    
    mo.md(f"**🔍 Checking file:** `{CSV_FILE_PATH}`")
    
    try:
        if os.path.exists(CSV_FILE_PATH):
            data_df = pd.read_csv(CSV_FILE_PATH)
            
            # Add review columns if missing
            review_columns = ['reviewer_initials', 'review_timestamp', 'model_prediction_plausible', 
                             'better_sic_available', 'recommended_sic_code', 'reviewer_notes']
            
            for col in review_columns:
                if col not in data_df.columns:
                    if col in ['model_prediction_plausible', 'better_sic_available']:
                        data_df[col] = None
                    else:
                        data_df[col] = ""
            
            entry_count = len(data_df)
            review_count = sum(data_df['reviewer_initials'].notna() & (data_df['reviewer_initials'] != ""))
            
            mo.md(f"""
            ## ✅ Data Loaded Successfully!
            
            - **📊 Total entries:** {entry_count}
            - **✅ Already reviewed:** {review_count}  
            - **📋 Columns found:** {len(data_df.columns)}
            
            **🔍 Sample from first entry:**
            - Job Title: `{data_df.iloc[0].get('soc2020_job_title', 'N/A')}`
            - Industry: `{data_df.iloc[0].get('sic2007_employee', 'N/A')}`
            - Model SIC: `{data_df.iloc[0].get('final_sic', 'N/A')}`
            """)
        else:
            mo.md(f"❌ **File not found at:** `{CSV_FILE_PATH}`\n\nPlease check the file path and update the CSV_FILE_PATH variable.")
            
    except Exception as e:
        mo.md(f"❌ **Error loading file:** {str(e)}")
    
    return data_df,


@app.cell
def __(data_df, mo):
    # Navigation slider
    if data_df is not None:
        entry_slider = mo.ui.slider(
            start=1,
            stop=len(data_df),
            value=1,
            label=f"📍 Select Entry (1 to {len(data_df)}):",
            show_value=True
        )
        
        mo.md(f"""
        ## 🧭 Navigation
        
        Use the slider below to navigate between entries:
        
        {entry_slider}
        """)
    else:
        entry_slider = None
        mo.md("⚠️ **No navigation available** - Please load data first")
    
    return entry_slider,


@app.cell
def __(data_df, entry_slider, mo, pd, json, ast):
    # Display current entry
    if data_df is not None and entry_slider is not None:
        current_idx = entry_slider.value - 1
        current_row = data_df.iloc[current_idx]
        
        # Basic information
        job_title = current_row.get('soc2020_job_title', 'N/A')
        industry = current_row.get('sic2007_employee', 'N/A')
        model_sic = current_row.get('final_sic', 'N/A')
        job_desc = str(current_row.get('soc2020_job_description', 'N/A'))
        followup_q = current_row.get('follow-up_q', 'N/A')
        followup_a = current_row.get('answer_to_followup', 'N/A')
        
        # Parse SIC candidates
        candidates_display = "No candidates available"
        try:
            candidates_raw = current_row.get('sic_candidates', '')
            if pd.notna(candidates_raw) and candidates_raw != "":
                # Try to parse as JSON
                if isinstance(candidates_raw, str):
                    candidates_raw = candidates_raw.replace("'", '"')
                    parsed_data = json.loads(candidates_raw)
                else:
                    parsed_data = candidates_raw
                
                # Extract candidates list
                if isinstance(parsed_data, dict) and 'alt_candidates' in parsed_data:
                    candidates_list = parsed_data['alt_candidates']
                elif isinstance(parsed_data, list):
                    candidates_list = parsed_data
                else:
                    candidates_list = []
                
                # Format candidates for display
                if candidates_list:
                    candidates_display = ""
                    for i, candidate in enumerate(candidates_list, 1):
                        code = candidate.get('class_code', 'N/A')
                        description = candidate.get('class_descriptive', 'N/A')
                        likelihood = candidate.get('likelihood', 0)
                        
                        # Emoji based on likelihood
                        if likelihood >= 0.7:
                            emoji = "🟢"
                        elif likelihood >= 0.4:
                            emoji = "🟡"
                        else:
                            emoji = "🔴"
                        
                        candidates_display += f"**{i}.** {emoji} **{code}** - {description}\n"
                        candidates_display += f"   *Likelihood: {likelihood:.1%}*\n\n"
        except Exception as e:
            candidates_display = f"Error parsing candidates: {str(e)}"
        
        # Review status
        has_review = pd.notna(current_row.get('reviewer_initials')) and current_row.get('reviewer_initials') != ""
        status_emoji = "🟢" if has_review else "🔴"
        status_text = "REVIEWED" if has_review else "NOT REVIEWED"
        
        mo.md(f"""
        ## {status_emoji} Entry {current_idx + 1} of {len(data_df)} - {status_text}
        
        ### 📋 Job Information
        **Job Title:** {job_title}  
        **Industry:** {industry}  
        **Model SIC Code:** `{model_sic}`
        
        ### 📄 Job Description
        {job_desc[:500]}{"..." if len(job_desc) > 500 else ""}
        
        ### 💬 Follow-up Information
        **Question:** {followup_q}  
        **Answer:** {followup_a}
        
        ### 📊 SIC Code Candidates
        {candidates_display}
        
        ### 📝 Existing Review
        **Reviewer:** {current_row.get('reviewer_initials', 'None')}  
        **Model Plausible:** {current_row.get('model_prediction_plausible', 'Not answered')}  
        **Better Available:** {current_row.get('better_sic_available', 'Not answered')}  
        **Recommended SIC:** {current_row.get('recommended_sic_code', 'None')}  
        **Notes:** {current_row.get('reviewer_notes', 'None')}
        
        ---
        **Use the form below to add or update your review:**
        """)
    else:
        mo.md("⚠️ **No entry to display** - Please check that data is loaded and an entry is selected")
    
    return


@app.cell
def __(mo):
    # Review input form
    mo.md("## 📝 Review Form")
    
    reviewer_initials = mo.ui.text(
        label="Your Initials:",
        placeholder="e.g., JD"
    )
    
    model_plausible = mo.ui.radio(
        options=["Yes", "No"],
        label="Is the model prediction plausible?"
    )
    
    better_available = mo.ui.radio(
        options=["Yes", "No"],
        label="Is a better SIC code available?"
    )
    
    recommended_sic = mo.ui.text(
        label="Recommended SIC Code (if applicable):",
        placeholder="Enter SIC code if you have a better suggestion"
    )
    
    review_notes = mo.ui.text_area(
        label="Notes:",
        placeholder="Any additional comments or observations"
    )
    
    submit_review = mo.ui.button(
        label="💾 Save Review",
        kind="success"
    )
    
    mo.vstack([
        reviewer_initials,
        model_plausible,
        better_available,
        recommended_sic,
        review_notes,
        submit_review
    ])
    return (
        better_available,
        model_plausible,
        recommended_sic,
        review_notes,
        reviewer_initials,
        submit_review,
    )


@app.cell
def __(
    submit_review,
    reviewer_initials,
    model_plausible,
    better_available,
    data_df,
    entry_slider,
    recommended_sic,
    review_notes,
    mo,
    datetime,
):
    # Handle review submission
    if submit_review.value and data_df is not None and entry_slider is not None:
        # Validation
        if not reviewer_initials.value or not reviewer_initials.value.strip():
            mo.md("❌ **Please enter your initials**")
        elif not model_plausible.value:
            mo.md("❌ **Please answer whether the model prediction is plausible**")
        elif not better_available.value:
            mo.md("❌ **Please answer whether a better SIC code is available**")
        else:
            # Save the review
            review_idx = entry_slider.value - 1
            data_df.loc[review_idx, 'reviewer_initials'] = reviewer_initials.value.strip()
            data_df.loc[review_idx, 'review_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data_df.loc[review_idx, 'model_prediction_plausible'] = (model_plausible.value == "Yes")
            data_df.loc[review_idx, 'better_sic_available'] = (better_available.value == "Yes")
            data_df.loc[review_idx, 'recommended_sic_code'] = recommended_sic.value.strip() if recommended_sic.value else ""
            data_df.loc[review_idx, 'reviewer_notes'] = review_notes.value.strip() if review_notes.value else ""
            
            mo.md(f"✅ **Review saved successfully for entry {entry_slider.value}!**")
    
    return


@app.cell
def __(data_df, mo):
    # Export button
    if data_df is not None:
        download_button = mo.ui.button(
            label="📥 Download Results CSV",
            kind="neutral"
        )
        
        mo.md(f"""
        ## 💾 Export Results
        
        Click the button below to download your reviewed data:
        
        {download_button}
        """)
    else:
        download_button = None
        mo.md("")
    
    return download_button,


@app.cell
def __(download_button, data_df, mo, datetime):
    # Handle download
    if download_button is not None and download_button.value and data_df is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_content = data_df.to_csv(index=False)
        
        mo.download(
            data=csv_content.encode('utf-8'),
            filename=f"sic_review_results_{timestamp}.csv",
            mimetype="text/csv"
        )
        
        mo.md("📥 **Download started!** Check your downloads folder.")
    
    return


@app.cell
def __(data_df, mo):
    # Progress summary
    if data_df is not None:
        total_entries = len(data_df)
        completed = sum(data_df['reviewer_initials'].notna() & (data_df['reviewer_initials'] != ""))
        progress_percent = (completed / total_entries * 100) if total_entries > 0 else 0
        
        # Text-based progress bar
        filled_blocks = int(progress_percent / 5)
        progress_bar = "█" * filled_blocks + "░" * (20 - filled_blocks)
        
        mo.md(f"""
        ## 📊 Review Progress
        
        **Completed:** {completed} / {total_entries} entries ({progress_percent:.1f}%)
        
        `{progress_bar}` {progress_percent:.1f}%
        
        ### Quick Stats:
        - **Remaining:** {total_entries - completed} entries
        - **Average per day:** Complete ~50 entries to finish in 2-3 weeks
        """)
    else:
        mo.md("")
    
    return


if __name__ == "__main__":
    app.run()