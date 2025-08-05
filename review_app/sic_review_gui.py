#!/usr/bin/env python3
"""
SIC Code Review Tool - Tkinter GUI Version
A desktop GUI application for reviewing SIC code predictions
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import json
import ast
import os
from datetime import datetime
import sys

class SICReviewGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SIC Code Review Tool")
        self.root.geometry("1200x800")
        
        # Data variables
        self.df = None
        self.current_index = 0
        self.filename = None
        
        # GUI variables
        self.reviewer_initials = tk.StringVar()
        self.model_plausible = tk.StringVar(value="")
        self.better_available = tk.StringVar(value="")
        self.recommended_sic = tk.StringVar()
        self.notes = tk.StringVar()
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # File upload section
        self.setup_file_section(main_frame)
        
        # Progress section
        self.setup_progress_section(main_frame)
        
        # Main content area
        self.setup_content_section(main_frame)
        
        # Navigation section
        self.setup_navigation_section(main_frame)
        
        # Review section
        self.setup_review_section(main_frame)
        
        # Save section
        self.setup_save_section(main_frame)
    
    def setup_file_section(self, parent):
        """Setup file upload section"""
        file_frame = ttk.LabelFrame(parent, text="File Upload", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Button(file_frame, text="Load CSV File", command=self.load_file).grid(row=0, column=0, padx=(0, 10))
        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
    
    def setup_progress_section(self, parent):
        """Setup progress tracking section"""
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="5")
        progress_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_frame.columnconfigure(3, weight=1)
        
        self.total_label = ttk.Label(progress_frame, text="Total: 0")
        self.total_label.grid(row=0, column=0, padx=(0, 10))
        
        self.reviewed_label = ttk.Label(progress_frame, text="Reviewed: 0")
        self.reviewed_label.grid(row=0, column=1, padx=(0, 10))
        
        self.remaining_label = ttk.Label(progress_frame, text="Remaining: 0")
        self.remaining_label.grid(row=0, column=2, padx=(0, 10))
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.grid(row=0, column=3, sticky=(tk.W, tk.E), padx=(10, 0))
    
    def setup_content_section(self, parent):
        """Setup main content display section"""
        content_frame = ttk.Frame(parent)
        content_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Left panel - Job Information
        left_frame = ttk.LabelFrame(content_frame, text="Job Information", padding="5")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(3, weight=1)
        
        # Job details
        self.job_title_label = ttk.Label(left_frame, text="Job Title: ", wraplength=300)
        self.job_title_label.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.industry_label = ttk.Label(left_frame, text="Industry: ", wraplength=300)
        self.industry_label.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Label(left_frame, text="Job Description:").grid(row=2, column=0, sticky=(tk.W), pady=(0, 5))
        self.job_desc_text = scrolledtext.ScrolledText(left_frame, height=8, wrap=tk.WORD)
        self.job_desc_text.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Follow-up information
        followup_frame = ttk.LabelFrame(left_frame, text="Follow-up", padding="5")
        followup_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        followup_frame.columnconfigure(0, weight=1)
        
        self.followup_q_label = ttk.Label(followup_frame, text="Question: ", wraplength=300)
        self.followup_q_label.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.followup_a_label = ttk.Label(followup_frame, text="Answer: ", wraplength=300)
        self.followup_a_label.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Right panel - SIC Information
        right_frame = ttk.LabelFrame(content_frame, text="SIC Information", padding="5")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(2, weight=1)
        
        # Model prediction
        self.model_pred_label = ttk.Label(right_frame, text="Model Prediction: ", font=('TkDefaultFont', 10, 'bold'))
        self.model_pred_label.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # SIC Candidates
        ttk.Label(right_frame, text="SIC Candidates:", font=('TkDefaultFont', 10, 'bold')).grid(row=1, column=0, sticky=(tk.W), pady=(0, 5))
        
        # Create treeview for candidates
        columns = ('Code', 'Description', 'Likelihood')
        self.candidates_tree = ttk.Treeview(right_frame, columns=columns, show='headings', height=10)
        
        # Configure columns
        self.candidates_tree.heading('Code', text='Code')
        self.candidates_tree.heading('Description', text='Description')
        self.candidates_tree.heading('Likelihood', text='Likelihood')
        
        self.candidates_tree.column('Code', width=80)
        self.candidates_tree.column('Description', width=250)
        self.candidates_tree.column('Likelihood', width=80)
        
        # Scrollbar for treeview
        candidates_scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.candidates_tree.yview)
        self.candidates_tree.configure(yscrollcommand=candidates_scrollbar.set)
        
        self.candidates_tree.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        candidates_scrollbar.grid(row=2, column=1, sticky=(tk.N, tk.S))
        
        right_frame.columnconfigure(0, weight=1)
    
    def setup_navigation_section(self, parent):
        """Setup navigation controls"""
        nav_frame = ttk.LabelFrame(parent, text="Navigation", padding="5")
        nav_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(nav_frame, text="⬅ Previous", command=self.previous_entry).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(nav_frame, text="Next ➡", command=self.next_entry).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Label(nav_frame, text="Jump to:").grid(row=0, column=2, padx=(0, 5))
        self.jump_entry = ttk.Entry(nav_frame, width=10)
        self.jump_entry.grid(row=0, column=3, padx=(0, 5))
        ttk.Button(nav_frame, text="Go", command=self.jump_to_entry).grid(row=0, column=4, padx=(0, 10))
        
        ttk.Button(nav_frame, text="Next Unreviewed", command=self.next_unreviewed).grid(row=0, column=5, padx=(0, 10))
        
        self.current_entry_label = ttk.Label(nav_frame, text="Entry: 0 / 0")
        self.current_entry_label.grid(row=0, column=6, padx=(10, 0))
    
    def setup_review_section(self, parent):
        """Setup review input section"""
        review_frame = ttk.LabelFrame(parent, text="Review Entry", padding="5")
        review_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        review_frame.columnconfigure(1, weight=1)
        review_frame.columnconfigure(3, weight=1)
        
        # Reviewer initials
        ttk.Label(review_frame, text="Reviewer Initials:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Entry(review_frame, textvariable=self.reviewer_initials, width=15).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 20))
        
        # Model plausible
        ttk.Label(review_frame, text="Model prediction plausible:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        plausible_frame = ttk.Frame(review_frame)
        plausible_frame.grid(row=0, column=3, sticky=(tk.W, tk.E))
        ttk.Radiobutton(plausible_frame, text="Yes", variable=self.model_plausible, value="True").grid(row=0, column=0, padx=(0, 10))
        ttk.Radiobutton(plausible_frame, text="No", variable=self.model_plausible, value="False").grid(row=0, column=1)
        
        # Better available
        ttk.Label(review_frame, text="Better SIC available:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(10, 0))
        better_frame = ttk.Frame(review_frame)
        better_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(10, 0))
        ttk.Radiobutton(better_frame, text="Yes", variable=self.better_available, value="True").grid(row=0, column=0, padx=(0, 10))
        ttk.Radiobutton(better_frame, text="No", variable=self.better_available, value="False").grid(row=0, column=1)
        
        # Recommended SIC
        ttk.Label(review_frame, text="Recommended SIC:").grid(row=1, column=2, sticky=tk.W, padx=(20, 5), pady=(10, 0))
        ttk.Entry(review_frame, textvariable=self.recommended_sic, width=20).grid(row=1, column=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Notes
        ttk.Label(review_frame, text="Notes:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5), pady=(10, 0))
        notes_entry = ttk.Entry(review_frame, textvariable=self.notes, width=50)
        notes_entry.grid(row=2, column=1, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Save button
        ttk.Button(review_frame, text="💾 Save Review", command=self.save_review).grid(row=3, column=0, columnspan=4, pady=(15, 0))
    
    def setup_save_section(self, parent):
        """Setup save/export section"""
        save_frame = ttk.LabelFrame(parent, text="Export Data", padding="5")
        save_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        ttk.Button(save_frame, text="📊 Export Excel", command=lambda: self.export_data('excel')).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(save_frame, text="📄 Export CSV", command=lambda: self.export_data('csv')).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(save_frame, text="📋 Export JSON", command=lambda: self.export_data('json')).grid(row=0, column=2)
    
    def load_file(self):
        """Load CSV file"""
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.df = pd.read_csv(filename)
                self.filename = filename
                
                # Add review columns if they don't exist
                review_columns = {
                    'reviewer_initials': "",
                    'review_timestamp': "",
                    'model_prediction_plausible': None,
                    'better_sic_available': None,
                    'recommended_sic_code': "",
                    'reviewer_notes': ""
                }
                
                for col, default_value in review_columns.items():
                    if col not in self.df.columns:
                        self.df[col] = default_value
                
                self.current_index = 0
                self.update_display()
                self.file_label.config(text=f"Loaded: {os.path.basename(filename)} ({len(self.df)} entries)")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def parse_sic_candidates(self, candidates_str):
        """Parse SIC candidates from string format"""
        if pd.isna(candidates_str) or candidates_str == "":
            return []
        
        try:
            # Try parsing as JSON first
            if isinstance(candidates_str, str):
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
    
    def is_reviewed(self, row):
        """Check if a row has been reviewed"""
        return pd.notna(row['reviewer_initials']) and row['reviewer_initials'] != ""
    
    def update_display(self):
        """Update the display with current entry data"""
        if self.df is None or self.df.empty:
            return
        
        if self.current_index >= len(self.df):
            self.current_index = 0
        
        row = self.df.iloc[self.current_index]
        
        # Update job information
        self.job_title_label.config(text=f"Job Title: {row.get('soc2020_job_title', 'N/A')}")
        self.industry_label.config(text=f"Industry: {row.get('sic2007_employee', 'N/A')}")
        
        # Update job description
        self.job_desc_text.delete(1.0, tk.END)
        self.job_desc_text.insert(1.0, str(row.get('soc2020_job_description', 'N/A')))
        
        # Update follow-up information
        self.followup_q_label.config(text=f"Question: {row.get('follow-up_q', 'N/A')}")
        self.followup_a_label.config(text=f"Answer: {row.get('answer_to_followup', 'N/A')}")
        
        # Update model prediction
        self.model_pred_label.config(text=f"Model Prediction: {row.get('final_sic', 'N/A')}")
        
        # Update SIC candidates
        self.candidates_tree.delete(*self.candidates_tree.get_children())
        candidates = self.parse_sic_candidates(row.get('sic_candidates', ''))
        
        for candidate in candidates:
            likelihood = candidate.get('likelihood', 0)
            code = candidate.get('class_code', 'N/A')
            description = candidate.get('class_descriptive', 'N/A')
            
            # Add likelihood indicator
            if likelihood >= 0.7:
                indicator = "🟢"
            elif likelihood >= 0.4:
                indicator = "🟡"
            else:
                indicator = "🔴"
            
            self.candidates_tree.insert('', 'end', values=(
                f"{indicator} {code}",
                description[:50] + "..." if len(description) > 50 else description,
                f"{likelihood:.1%}"
            ))
        
        # Update review fields with existing data, but preserve initials if already entered
        current_initials = self.reviewer_initials.get()
        if not current_initials.strip():
            # Only update initials if they're empty
            self.reviewer_initials.set(row.get('reviewer_initials', ''))
        
        # Handle boolean fields properly
        model_plausible = row.get('model_prediction_plausible')
        if pd.notna(model_plausible):
            self.model_plausible.set(str(model_plausible))
        else:
            self.model_plausible.set("")
        
        better_available = row.get('better_sic_available')
        if pd.notna(better_available):
            self.better_available.set(str(better_available))
        else:
            self.better_available.set("")
        
        self.recommended_sic.set(row.get('recommended_sic_code', ''))
        self.notes.set(row.get('reviewer_notes', ''))
        
        # Update progress
        self.update_progress()
        
        # Update navigation
        self.current_entry_label.config(text=f"Entry: {self.current_index + 1} / {len(self.df)}")
        
        # Update window title to show review status
        reviewed = self.is_reviewed(row)
        status = "🟢 REVIEWED" if reviewed else "🔴 NOT REVIEWED"
        self.root.title(f"SIC Code Review Tool - Entry {self.current_index + 1} - {status}")
    
    def update_progress(self):
        """Update progress indicators"""
        if self.df is None:
            return
        
        total = len(self.df)
        reviewed = sum(self.df.apply(self.is_reviewed, axis=1))
        remaining = total - reviewed
        progress = (reviewed / total * 100) if total > 0 else 0
        
        self.total_label.config(text=f"Total: {total}")
        self.reviewed_label.config(text=f"Reviewed: {reviewed}")
        self.remaining_label.config(text=f"Remaining: {remaining}")
        self.progress_bar['value'] = progress
    
    def previous_entry(self):
        """Go to previous entry"""
        if self.df is None or self.current_index <= 0:
            return
        self.current_index -= 1
        self.update_display()
    
    def next_entry(self):
        """Go to next entry"""
        if self.df is None or self.current_index >= len(self.df) - 1:
            return
        self.current_index += 1
        self.update_display()
    
    def jump_to_entry(self):
        """Jump to specific entry"""
        if self.df is None:
            return
        
        try:
            entry_num = int(self.jump_entry.get())
            if 1 <= entry_num <= len(self.df):
                self.current_index = entry_num - 1
                self.update_display()
                self.jump_entry.delete(0, tk.END)
            else:
                messagebox.showerror("Error", f"Entry number must be between 1 and {len(self.df)}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
    
    def next_unreviewed(self):
        """Go to next unreviewed entry"""
        if self.df is None:
            return
        
        for i in range(self.current_index + 1, len(self.df)):
            if not self.is_reviewed(self.df.iloc[i]):
                self.current_index = i
                self.update_display()
                return
        
        messagebox.showinfo("Info", "No more unreviewed entries found!")
    
    def save_review(self):
        """Save current review"""
        if self.df is None:
            return
        
        # Validate input
        if not self.reviewer_initials.get().strip():
            messagebox.showerror("Error", "Please enter reviewer initials")
            return
        
        if not self.model_plausible.get():
            messagebox.showerror("Error", "Please answer if model prediction is plausible")
            return
        
        if not self.better_available.get():
            messagebox.showerror("Error", "Please answer if better SIC code is available")
            return
        
        # Save the review
        self.df.loc[self.current_index, 'reviewer_initials'] = self.reviewer_initials.get().strip()
        self.df.loc[self.current_index, 'review_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.df.loc[self.current_index, 'model_prediction_plausible'] = self.model_plausible.get() == "True"
        self.df.loc[self.current_index, 'better_sic_available'] = self.better_available.get() == "True"
        self.df.loc[self.current_index, 'recommended_sic_code'] = self.recommended_sic.get().strip()
        self.df.loc[self.current_index, 'reviewer_notes'] = self.notes.get().strip()
        
        # Store initials for future use (keep them for next review)
        stored_initials = self.reviewer_initials.get().strip()
        
        messagebox.showinfo("Success", "Review saved successfully!")
        
        # Clear form fields except initials
        self.model_plausible.set("")
        self.better_available.set("")
        self.recommended_sic.set("")
        self.notes.set("")
        
        # Keep the initials for next review
        self.reviewer_initials.set(stored_initials)
        
        # Auto-advance to next unreviewed entry
        for i in range(self.current_index + 1, len(self.df)):
            if not self.is_reviewed(self.df.iloc[i]):
                self.current_index = i
                self.update_display()
                return
        
        # If no more unreviewed entries, just update display
        self.update_display()
    
    def export_data(self, format_type):
        """Export data to file"""
        if self.df is None:
            messagebox.showerror("Error", "No data to export")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            if format_type == 'excel':
                filename = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                    title="Save Excel File"
                )
                if filename:
                    # Add timestamp to filename if not already there
                    if not any(char.isdigit() for char in filename):
                        base, ext = os.path.splitext(filename)
                        filename = f"{base}_{timestamp}{ext}"
                    
                    try:
                        self.df.to_excel(filename, index=False)
                        messagebox.showinfo("Success", f"Data exported to {filename}")
                    except ImportError:
                        messagebox.showerror("Error", "openpyxl is required for Excel export. Try CSV instead.")
                    except PermissionError:
                        messagebox.showerror("Error", f"Permission denied. Cannot write to {filename}")
                    except Exception as e:
                        messagebox.showerror("Error", f"Excel export failed: {str(e)}")
            
            elif format_type == 'csv':
                filename = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                    title="Save CSV File"
                )
                if filename:
                    # Add timestamp to filename if not already there
                    if not any(char.isdigit() for char in filename):
                        base, ext = os.path.splitext(filename)
                        filename = f"{base}_{timestamp}{ext}"
                    
                    try:
                        self.df.to_csv(filename, index=False)
                        messagebox.showinfo("Success", f"Data exported to {filename}")
                    except PermissionError:
                        messagebox.showerror("Error", f"Permission denied. Cannot write to {filename}")
                    except Exception as e:
                        messagebox.showerror("Error", f"CSV export failed: {str(e)}")
            
            elif format_type == 'json':
                filename = filedialog.asksaveasfilename(
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                    title="Save JSON File"
                )
                if filename:
                    # Add timestamp to filename if not already there
                    if not any(char.isdigit() for char in filename):
                        base, ext = os.path.splitext(filename)
                        filename = f"{base}_{timestamp}{ext}"
                    
                    try:
                        self.df.to_json(filename, orient='records', indent=2)
                        messagebox.showinfo("Success", f"Data exported to {filename}")
                    except PermissionError:
                        messagebox.showerror("Error", f"Permission denied. Cannot write to {filename}")
                    except Exception as e:
                        messagebox.showerror("Error", f"JSON export failed: {str(e)}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")


def main():
    root = tk.Tk()
    app = SICReviewGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()