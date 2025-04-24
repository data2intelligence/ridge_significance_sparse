=============
Visualization
=============

This page provides examples of how to visualize results from RidgeInference analyses.

Basic Visualization Functions
=============================

RidgeInference provides built-in visualization through the ``utils`` module:

.. code-block:: python

    from ridge_inference.utils import visualize_activity
    
    # Assume we have results from secact_inference
    result = {...}  # Output from secact_inference
    
    # Create visualization of top 20 proteins
    fig = visualize_activity(result, top_n=20, pvalue_threshold=0.05)
    
    # Save the figure
    fig.savefig("activity_visualization.png", dpi=300, bbox_inches="tight")

Customizing Built-in Visualizations
===================================

You can customize the built-in visualizations after creating them:

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    from ridge_inference.utils import visualize_activity
    
    # Create the basic visualization
    fig = visualize_activity(result, top_n=15, pvalue_threshold=0.05)
    
    # Get the current axes
    ax = plt.gca()
    
    # Customize appearance
    ax.set_title("Differential Protein Activity", fontsize=16, fontweight='bold')
    ax.set_xlabel("Activity Score", fontsize=12)
    ax.set_ylabel("Protein", fontsize=12)
    
    # Add a vertical line at zero
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    
    # Color bars based on activity direction
    bars = ax.patches
    for i, bar in enumerate(bars):
        # Get the current activity value
        activity = bar.get_width()
        if activity > 0:
            bar.set_facecolor('indianred')
        else:
            bar.set_facecolor('steelblue')
    
    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("customized_activity.png", dpi=300, bbox_inches="tight")

Creating Heatmaps
=================

Visualizing activity across multiple samples with heatmaps:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Assume we have results from secact_inference with multiple samples
    activities = result['beta']    # Protein × Sample DataFrame
    pvalues = result['pvalue']     # Protein × Sample DataFrame
    
    # Filter for significant proteins (in at least one sample)
    significant = (pvalues < 0.05).any(axis=1)
    sig_activities = activities.loc[significant]
    
    # If we have too many proteins, take the top N by absolute activity
    if len(sig_activities) > 30:
        # Calculate mean absolute activity across samples
        mean_abs = sig_activities.abs().mean(axis=1)
        # Get top 30 proteins
        top_proteins = mean_abs.nlargest(30).index
        sig_activities = sig_activities.loc[top_proteins]
    
    # Create clustered heatmap
    plt.figure(figsize=(12, 10))
    
    # Cluster rows (proteins) but not columns (samples)
    g = sns.clustermap(
        sig_activities,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        col_cluster=False,
        row_cluster=True,
        vmin=-2, vmax=2,  # Limit color scale
        cbar_kws={"label": "Activity Score"},
        dendrogram_ratio=(0.2, 0.05),
        figsize=(12, 10)
    )
    
    # Customize
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    g.ax_heatmap.set_ylabel("Protein", fontsize=12)
    g.ax_heatmap.set_xlabel("Sample", fontsize=12)
    g.fig.suptitle("Protein Activity Heatmap", fontsize=16, y=1.02)
    
    # Save figure
    plt.savefig("activity_heatmap.png", dpi=300, bbox_inches="tight")

Volcano Plots
=============

Creating volcano plots to visualize both activity and significance:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Assume we have results from secact_inference with a single comparison
    # If multiple samples, take the first one
    if activities.shape[1] > 1:
        activities_single = activities.iloc[:, 0]
        pvalues_single = pvalues.iloc[:, 0]
        sample_name = activities.columns[0]
    else:
        activities_single = activities.iloc[:, 0]
        pvalues_single = pvalues.iloc[:, 0]
        sample_name = "Comparison"
    
    # Create DataFrame for plotting
    volcano_data = pd.DataFrame({
        "Activity": activities_single,
        "P-value": pvalues_single,
        "-log10(P-value)": -np.log10(pvalues_single)
    })
    
    # Add significance flag
    volcano_data["Significant"] = pvalues_single < 0.05
    
    # Add protein labels
    volcano_data["Protein"] = volcano_data.index
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Base scatter plot
    sns.scatterplot(
        data=volcano_data,
        x="Activity",
        y="-log10(P-value)",
        hue="Significant",
        palette={True: "firebrick", False: "gray"},
        alpha=0.7,
        s=50
    )
    
    # Add threshold line
    plt.axhline(y=-np.log10(0.05), color="navy", linestyle="--", alpha=0.5, 
                label="P-value = 0.05")
    
    # Add labels to significant points (top 10 by p-value)
    top_sig = volcano_data[volcano_data["Significant"]].nlargest(10, "-log10(P-value)")
    
    for _, row in top_sig.iterrows():
        plt.text(
            row["Activity"],
            row["-log10(P-value)"] + 0.1,
            row["Protein"],
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1)
        )
    
    # Customize
    plt.title(f"Volcano Plot - {sample_name}", fontsize=14)
    plt.xlabel("Activity Score", fontsize=12)
    plt.ylabel("-log10(P-value)", fontsize=12)
    plt.legend(title="P-value < 0.05")
    plt.grid(True, linestyle="--", alpha=0.3)
    
    # Save
    plt.tight_layout()
    plt.savefig("volcano_plot.png", dpi=300)

Network Visualization
=====================

Creating network visualizations with protein interactions:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.colors import LinearSegmentedColormap
    
    # Assume we have results from secact_inference
    activities = result['beta']
    pvalues = result['pvalue']
    
    # If multiple samples, take the first one
    if activities.shape[1] > 1:
        activities_single = activities.iloc[:, 0]
        pvalues_single = pvalues.iloc[:, 0]
    else:
        activities_single = activities.iloc[:, 0]
        pvalues_single = pvalues.iloc[:, 0]
    
    # Filter for significant proteins
    significant = pvalues_single < 0.05
    sig_activities = activities_single[significant]
    
    # Take top proteins by absolute activity
    top_proteins = sig_activities.abs().nlargest(15).index
    top_activities = sig_activities[top_proteins]
    
    # Create a correlation matrix (placeholder for actual interactions)
    # In a real scenario, you would import actual protein interaction data
    np.random.seed(42)  # For reproducible example
    n_proteins = len(top_proteins)
    interactions = pd.DataFrame(
        np.random.normal(0, 0.3, (n_proteins, n_proteins)) + 
        np.eye(n_proteins) * 0.5,
        index=top_proteins,
        columns=top_proteins
    )
    
    # Create network
    G = nx.Graph()
    
    # Add nodes with attributes
    for protein in top_proteins:
        G.add_node(
            protein,
            activity=float(top_activities[protein]),
            size=abs(float(top_activities[protein])) * 500
        )
    
    # Add edges with weights
    for p1 in top_proteins:
        for p2 in top_proteins:
            if p1 != p2 and abs(interactions.loc[p1, p2]) > 0.3:
                G.add_edge(p1, p2, weight=abs(float(interactions.loc[p1, p2])) * 3)
    
    # Create plot
    plt.figure(figsize=(12, 12))
    
    # Position nodes using force-directed layout
    pos = nx.spring_layout(G, seed=42)
    
    # Custom colormap
    cmap = LinearSegmentedColormap.from_list(
        "activity", 
        [(0, "steelblue"), (0.5, "white"), (1, "firebrick")]
    )
    
    # Get node colors
    activities_array = np.array([G.nodes[p]["activity"] for p in G.nodes()])
    norm = plt.Normalize(-max(abs(activities_array)), max(abs(activities_array)))
    node_colors = cmap(norm(activities_array))
    
    # Draw network
    # Nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=[G.nodes[p]["size"] for p in G.nodes()],
        node_color=node_colors,
        alpha=0.8,
        edgecolors="black",
        linewidths=1
    )
    
    # Edges
    nx.draw_networkx_edges(
        G, pos,
        width=[G[u][v]["weight"] for u, v in G.edges()],
        alpha=0.5,
        edge_color="gray"
    )
    
    # Labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_family="sans-serif",
        font_weight="bold"
    )
    
    # Title and finalization
    plt.title("Protein Interaction Network", fontsize=16)
    plt.axis("off")
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation="vertical", pad=0.05)
    cbar.set_label("Activity Score", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("protein_network.png", dpi=300, bbox_inches="tight")

Interactive Visualization with Plotly
=====================================

Creating interactive visualizations with Plotly:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Assume we have results from secact_inference with multiple samples
    activities = result['beta']
    pvalues = result['pvalue']
    
    # Prepare data for plotting
    plot_data = pd.DataFrame()
    
    # Reshape from wide to long format
    for sample in activities.columns:
        sample_data = pd.DataFrame({
            "Protein": activities.index,
            "Activity": activities[sample],
            "P-value": pvalues[sample],
            "Sample": sample,
            "-log10(P)": -np.log10(pvalues[sample]),
            "Significant": pvalues[sample] < 0.05
        })
        plot_data = pd.concat([plot_data, sample_data])
    
    # Create interactive bar chart
    fig = px.bar(
        plot_data[plot_data["Significant"]].nlargest(20, "Activity"),
        x="Activity",
        y="Protein",
        color="Activity",
        color_continuous_scale="RdBu_r",
        facet_col="Sample",
        hover_data=["P-value"],
        height=600,
        title="Top Active Proteins by Sample",
        orientation="h"
    )
    
    # Customize
    fig.update_layout(
        xaxis_title="Activity Score",
        yaxis_title="Protein",
        coloraxis_colorbar_title="Activity"
    )
    
    # Save as HTML
    fig.write_html("interactive_activity.html")
    
    # Create interactive volcano plot
    volcano_fig = px.scatter(
        plot_data,
        x="Activity",
        y="-log10(P)",
        color="Significant",
        color_discrete_map={True: "red", False: "gray"},
        hover_data=["Protein", "Sample", "P-value"],
        facet_col="Sample",
        title="Volcano Plot by Sample"
    )
    
    # Add threshold line
    for i in range(len(activities.columns)):
        volcano_fig.add_shape(
            type="line",
            x0=volcano_fig.data[i].x.min(),
            x1=volcano_fig.data[i].x.max(),
            y0=-np.log10(0.05),
            y1=-np.log10(0.05),
            line=dict(color="blue", width=1, dash="dash"),
            xref=f"x{i+1}" if i > 0 else "x",
            yref=f"y{i+1}" if i > 0 else "y"
        )
    
    # Customize
    volcano_fig.update_layout(
        xaxis_title="Activity Score",
        yaxis_title="-log10(P-value)"
    )
    
    # Save as HTML
    volcano_fig.write_html("interactive_volcano.html")
