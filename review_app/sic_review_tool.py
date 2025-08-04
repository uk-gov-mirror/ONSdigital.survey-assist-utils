import streamlit as st
import pandas as pd
import json
import ast
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="SIC Code Review Tool",
    page_icon="📊",
    layout="wide"
)

def parse_sic_candidates(candidates_str):
    """Parse SIC candidates from string format"""
    if pd.isna(candidates_str) or candidates_str == "":
        return []
    
    try:
        # Try parsing as JSON first
        if isinstance(candidates_str, str):
            # Handle potential single quotes vs double quotes
            candidates_str = candidates_str.replace("'", '"')
            data = json.loads(candidates_str)
        else:
            data = candidates_str
            
        if isinstance(data, dict) and 'alt_candidates' in data:
            return data['alt_candidates']
        elif isinstance(data, list):
            return data
        else:
            return []
    except:
        try:
            # Try evaluating as Python literal
            data = ast.literal_eval(str(candidates_str))
            if isinstance(data, dict) and 'alt_candidates' in data:
                return data['alt_candidates']
            elif isinstance(data, list):
                return data
            else:
                return []
        except:
            return []

def load_data(uploaded_file):
    """Load and prepare the Excel data"""
    df = pd.read_excel(uploaded_file)
    
    # Add review columns if they don't exist
    review_columns = [
        'reviewer_initials',
        'review_timestamp',
        'model_prediction_plausible',
        'better_sic_available',
        'recommended_sic_code',
        'reviewer_notes'
    ]
    
    for col in review_columns:
        if col not in df.columns:
            if col in ['model_prediction_plausible', 'better_sic_available']:
                df[col] = None
            else:
                df[col] = ""
    
    return df

def is_reviewed(row):
    """Check if a row has been reviewed"""
    return pd.notna(row['reviewer_initials']) and row['reviewer_initials'] != ""

def get_review_status_color(reviewed):
    """Get color for review status"""
    return "🟢" if reviewed else "🔴"

def save_dataframe_as_excel(df):
    """Convert dataframe to Excel bytes"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Reviews')
    return output.getvalue()

def save_dataframe_as_csv(df):
    """Convert dataframe to CSV bytes"""
    return df.to_csv(index=False).encode('utf-8')

def save_dataframe_as_json(df):
    """Convert dataframe to JSON bytes"""
    return df.to_json(orient='records', indent=2).encode('utf-8')

def main():
    st.title("📊 SIC Code Review Tool")
    st.markdown("Review model predictions for SIC code assignments")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Excel file", 
        type=['xlsx', 'xls'],
        help="Upload your Excel file containing the SIC code data to review"
    )
    
    if uploaded_file is not None:
        # Load data
        if 'df' not in st.session_state:
            st.session_state.df = load_data(uploaded_file)
            st.session_state.current_index = 0
        
        df = st.session_state.df
        total_rows = len(df)
        reviewed_count = sum(df.apply(is_reviewed, axis=1))
        
        # Progress indicator
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Entries", total_rows)
        with col2:
            st.metric("Reviewed", reviewed_count)
        with col3:
            st.metric("Remaining", total_rows - reviewed_count)
        
        progress = reviewed_count / total_rows if total_rows > 0 else 0
        st.progress(progress)
        
        # Navigation
        st.markdown("---")
        nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)
        
        with nav_col1:
            if st.button("⬅️ Previous", disabled=st.session_state.current_index <= 0):
                st.session_state.current_index = max(0, st.session_state.current_index - 1)
                st.rerun()
        
        with nav_col2:
            if st.button("➡️ Next", disabled=st.session_state.current_index >= total_rows - 1):
                st.session_state.current_index = min(total_rows - 1, st.session_state.current_index + 1)
                st.rerun()
        
        with nav_col3:
            # Jump to entry
            jump_to = st.number_input(
                "Jump to entry", 
                min_value=1, 
                max_value=total_rows, 
                value=st.session_state.current_index + 1
            )
            if st.button("Go"):
                st.session_state.current_index = jump_to - 1
                st.rerun()
        
        with nav_col4:
            if st.button("🔍 Next Unreviewed"):
                # Find next unreviewed entry
                for i in range(st.session_state.current_index + 1, total_rows):
                    if not is_reviewed(df.iloc[i]):
                        st.session_state.current_index = i
                        st.rerun()
                        break
                else:
                    st.info("No more unreviewed entries found")
        
        with nav_col5:
            # Show entry status
            current_reviewed = is_reviewed(df.iloc[st.session_state.current_index])
            status_color = get_review_status_color(current_reviewed)
            st.markdown(f"**Status:** {status_color} {'Reviewed' if current_reviewed else 'Not Reviewed'}")
        
        # Current entry info
        current_row = df.iloc[st.session_state.current_index]
        st.markdown(f"### Entry {st.session_state.current_index + 1} of {total_rows}")
        
        # Display current entry data
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Job Information")
            st.markdown(f"**Job Title:** {current_row.get('soc2020_job_title', 'N/A')}")
            st.markdown(f"**Industry:** {current_row.get('sic2007_employee', 'N/A')}")
            
            with st.expander("Job Description", expanded=False):
                st.write(current_row.get('soc2020_job_description', 'N/A'))
            
            st.markdown("#### Follow-up Information")
            st.markdown(f"**Question:** {current_row.get('follow-up_q', 'N/A')}")
            st.markdown(f"**Answer:** {current_row.get('answer_to_followup', 'N/A')}")
            
            st.markdown(f"#### Model Prediction")
            st.markdown(f"**Final SIC Code:** `{current_row.get('final_sic', 'N/A')}`")
        
        with col2:
            st.markdown("#### SIC Code Candidates")
            candidates = parse_sic_candidates(current_row.get('sic_candidates', ''))
            
            if candidates:
                for i, candidate in enumerate(candidates):
                    likelihood = candidate.get('likelihood', 0)
                    code = candidate.get('class_code', 'N/A')
                    description = candidate.get('class_descriptive', 'N/A')
                    
                    # Color code by likelihood
                    if likelihood >= 0.7:
                        color = "🟢"
                    elif likelihood >= 0.4:
                        color = "🟡"
                    else:
                        color = "🔴"
                    
                    st.markdown(f"{color} **{code}**: {description}")
                    st.markdown(f"   *Likelihood: {likelihood:.1%}*")
            else:
                st.write("No candidates available")
        
        # Review form
        st.markdown("---")
        st.markdown("### Review This Entry")
        
        form_col1, form_col2 = st.columns(2)
        
        with form_col1:
            # Pre-fill with existing values if available
            reviewer_initials = st.text_input(
                "Reviewer Initials",
                value=current_row.get('reviewer_initials', ''),
                max_chars=10
            )
            
            model_plausible = st.radio(
                "Model prediction is plausible",
                options=[None, True, False],
                format_func=lambda x: "Not answered" if x is None else ("Yes" if x else "No"),
                index=0 if pd.isna(current_row.get('model_prediction_plausible')) else (1 if current_row.get('model_prediction_plausible') else 2)
            )
            
            better_available = st.radio(
                "Better SIC code is available",
                options=[None, True, False],
                format_func=lambda x: "Not answered" if x is None else ("Yes" if x else "No"),
                index=0 if pd.isna(current_row.get('better_sic_available')) else (1 if current_row.get('better_sic_available') else 2)
            )
        
        with form_col2:
            recommended_sic = st.text_input(
                "Enter recommended SIC code (if applicable)",
                value=current_row.get('recommended_sic_code', ''),
                help="Enter the SIC code you recommend, or leave blank if model prediction is acceptable"
            )
            
            notes = st.text_area(
                "Notes",
                value=current_row.get('reviewer_notes', ''),
                height=100
            )
        
        # Save review
        if st.button("💾 Save Review", type="primary"):
            if reviewer_initials and model_plausible is not None and better_available is not None:
                # Update the dataframe
                st.session_state.df.loc[st.session_state.current_index, 'reviewer_initials'] = reviewer_initials
                st.session_state.df.loc[st.session_state.current_index, 'review_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.df.loc[st.session_state.current_index, 'model_prediction_plausible'] = model_plausible
                st.session_state.df.loc[st.session_state.current_index, 'better_sic_available'] = better_available
                st.session_state.df.loc[st.session_state.current_index, 'recommended_sic_code'] = recommended_sic
                st.session_state.df.loc[st.session_state.current_index, 'reviewer_notes'] = notes
                
                st.success("Review saved successfully!")
                
                # Auto-advance to next unreviewed entry
                for i in range(st.session_state.current_index + 1, total_rows):
                    if not is_reviewed(st.session_state.df.iloc[i]):
                        st.session_state.current_index = i
                        st.rerun()
                        break
            else:
                st.error("Please fill in reviewer initials and answer both plausibility questions.")
        
        # Download options
        st.markdown("---")
        st.markdown("### Download Results")
        
        download_col1, download_col2, download_col3 = st.columns(3)
        
        with download_col1:
            excel_data = save_dataframe_as_excel(st.session_state.df)
            st.download_button(
                label="📊 Download Excel",
                data=excel_data,
                file_name=f"sic_review_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with download_col2:
            csv_data = save_dataframe_as_csv(st.session_state.df)
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"sic_review_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with download_col3:
            json_data = save_dataframe_as_json(st.session_state.df)
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name=f"sic_review_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Show summary of all entries (optional)
        if st.checkbox("Show all entries summary"):
            st.markdown("### All Entries Overview")
            summary_df = st.session_state.df[['soc2020_job_title', 'final_sic', 'reviewer_initials', 'model_prediction_plausible', 'better_sic_available']].copy()
            summary_df['Status'] = summary_df.apply(lambda row: '🟢 Reviewed' if is_reviewed(row) else '🔴 Not Reviewed', axis=1)
            st.dataframe(summary_df, use_container_width=True, height=300)

if __name__ == "__main__":
    main()