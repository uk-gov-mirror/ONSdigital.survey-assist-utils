
'''Utility script to produce sankey diagram for SA codability and plausibility'''

import pandas as pd
import plotly.graph_objects as go

###
# Luke Notes
#
# 150 total:
############
# 90 CC Codable
# -> 73 SA codable, 17 SA Not Codable
# 60 CC Not Codable
# -> 28 SA Codable, 32 SA Not Codable
###
# Create dataframe directly from your provided data
data = [
    # CC Codable -> SA Codable -> Exact match/Plausible (62 total from SA Codable)
    *[['Codable', 'Codable', 'Exact match/Plausible']] * 62, # DONE
    
    # CC Not codable -> SA Codable -> Exact match/Plausible (remaining 15 from SA Codable)
    *[['Not codable', 'Codable', 'Exact match/Plausible']] * 11, 
    
    # CC Not codable -> SA Codable -> Not plausible (5 total from SA Codable)
    *[['Not codable', 'Codable', 'Not plausible']] * 17, # DONE
    
    # CC Codable -> SA Not codable -> Exact match/Plausible (27 total from SA Not codable)
    *[['Codable', 'Not codable', 'Exact match/Plausible']] * 11, # DONE
    
    # CC Not codable -> SA Not codable -> Exact match/Plausible (remaining 12 from SA Not codable)  
    *[['Not codable', 'Not codable', 'Exact match/Plausible']] * 21,
    
    # CC Not codable -> SA Not codable -> Not plausible (8 total from SA Not codable)
    *[['Not codable', 'Not codable', 'Not plausible']] * 11,

    # CC Codable -> SA Codable -> Not plausible (? total from SA Codable)
    *[['Codable', 'Codable', 'Not plausible']] * 11, 

    # CC Codable -> SA Codable -> Not plausible (? total from SA Codable)
    *[['Codable', 'Not codable', 'Not plausible']] * 6, 
]

# Create DataFrame
df = pd.DataFrame(data, columns=['CC_Decision', 'SA_Prediction', 'Agreement_Outcome'])

print("DataFrame shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nData summary:")
print(df.groupby(['CC_Decision', 'SA_Prediction', 'Agreement_Outcome']).size().reset_index(name='count'))

def create_sankey_from_df(df):
    """
    Create Sankey diagram from the dataframe
    """
    
    # Define nodes
    nodes = [
        "CC: Codable without<br>follow-up",           # 0
        "CC: Not codable without<br>follow-up",       # 1
        "SA: Codable without<br>follow-up",           # 2
        "SA: Not codable without<br>follow-up",       # 3
        "SA: Exact match or<br>plausible",            # 4
        "SA: Not plausible"                           # 5
    ]
    
    # Node colors
    node_colors = ['#ff7043', '#26a69a', '#42a5f5', '#5c6bc0', '#ab47bc', '#ef5350']
    
    # Calculate flows from your data
    cc_codable_total = len(df[df['CC_Decision'] == 'Codable'])
    cc_not_codable_total = len(df[df['CC_Decision'] == 'Not codable'])
    sa_codable_total = len(df[df['SA_Prediction'] == 'Codable'])
    sa_not_codable_total = len(df[df['SA_Prediction'] == 'Not codable'])
    exact_match_total = len(df[df['Agreement_Outcome'] == 'Exact match/Plausible'])
    not_plausible_total = len(df[df['Agreement_Outcome'] == 'Not plausible'])
    
    # Flow calculations
    flows = {
        'cc_codable_to_sa_codable': 62+11,
        'cc_codable_to_sa_not_codable': 11+6,
        'cc_not_codable_to_sa_codable': 11+17,
        'cc_not_codable_to_sa_not_codable': 21+11,
        'sa_codable_to_exact_match': 62+11,
        'sa_codable_to_not_plausible': 17+11,
        'sa_not_codable_to_exact_match': 11+21,
        'sa_not_codable_to_not_plausible': 11+6
    }
    
    # Add percentages to node labels
    total = len(df)
    nodes_with_data = [
        f"{nodes[0]}<br>{cc_codable_total} ({cc_codable_total*0.6666666:.1f}%)",
        f"{nodes[1]}<br>{cc_not_codable_total} ({cc_not_codable_total*0.6666666:.1f}%)",
        f"{nodes[2]}<br>{sa_codable_total} ({sa_codable_total*0.6666666:.1f}%)",
        f"{nodes[3]}<br>{sa_not_codable_total} ({sa_not_codable_total*0.6666666:.1f}%)",
        f"{nodes[4]}<br>{exact_match_total} ({exact_match_total*0.6666666:.1f}%)",
        f"{nodes[5]}<br>{not_plausible_total} ({not_plausible_total*0.6666666:.1f}%)"
    ]
    
    # Create links
    source_nodes = [0, 0, 1, 1, 2, 2, 3, 3]
    target_nodes = [2, 3, 2, 3, 4, 5, 4, 5]
    values = [
        flows['cc_codable_to_sa_codable'],
        flows['cc_codable_to_sa_not_codable'],
        flows['cc_not_codable_to_sa_codable'],
        flows['cc_not_codable_to_sa_not_codable'],
        flows['sa_codable_to_exact_match'],
        flows['sa_codable_to_not_plausible'],
        flows['sa_not_codable_to_exact_match'],
        flows['sa_not_codable_to_not_plausible']
    ]
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes_with_data,
            color=node_colors
        ),
        link=dict(
            source=source_nodes,
            target=target_nodes,
            value=values,
            color=['rgba(0,0,0,0.2)'] * len(values)
        )
    )])
    
    fig.update_layout(
        title="Clerical Coder vs Model Prediction Analysis<br><sub>n=150 cases</sub>",
        font_size=12,
        width=1000,
        height=600
    )
    
    return fig

# Create and show the diagram
fig = create_sankey_from_df(df)
fig.show()

# Print summary statistics
print(f"\n=== SUMMARY STATISTICS ===")
print(f"Total cases: {len(df)}")
print(f"CC Codable: {len(df[df['CC_Decision'] == 'Codable'])} (60%)")
print(f"CC Not codable: {len(df[df['CC_Decision'] == 'Not codable'])} (40%)")
print(f"SA Codable: {len(df[df['SA_Prediction'] == 'Codable'])} (65%)")
print(f"SA Not codable: {len(df[df['SA_Prediction'] == 'Not codable'])} (35%)")
print(f"SA Exact match/Plausible: {len(df[df['Agreement_Outcome'] == 'Exact match/Plausible'])} (87%)")
print(f"SA Not plausible: {len(df[df['Agreement_Outcome'] == 'Not plausible'])} (13%)")

# Save the dataframe
df.to_csv('coding_analysis_data.csv', index=False)
print(f"\nDataframe saved to 'coding_analysis_data.csv'")

# Save the figure
fig.write_html("sankey_diagram.html")
print("Sankey diagram saved to 'sankey_diagram.html'")