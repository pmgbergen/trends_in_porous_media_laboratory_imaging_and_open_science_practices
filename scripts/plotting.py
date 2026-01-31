import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import argparse

dpi = 1000
plt.rcParams.update({"font.size": 20})

parser = argparse.ArgumentParser(description="Visualize article analysis results")
parser.add_argument(
    "--input",
    "-i",
    type=str,
    required=True,
    help="Path to the input CSV file",
)
parser.add_argument(
    "--categories",
    "-c",
    type=str,
    nargs="+",
    default=["All"],
    help="List of categories to include in the analysis",
)
parser.add_argument(
    "--journal",
    type=str,
    required=True,
    help="Journal name to print in plots",
)
parser.add_argument(
    "--years",
    "-y",
    type=int,
    nargs="+",
    default=None,
)
parser.add_argument(
    "--horizontal-shift",
    type=float,
    default=0.03,
    help="Horizontal shift for title of grouped pie charts",
)
parser.add_argument(
    "--pie-size",
    "-p",
    type=float,
    default=1,
)
parser.add_argument(
    "--without-labels",
    action="store_true",
    help="Turn off labels in pie charts",
)
args = parser.parse_args()
categories = args.categories

# Load the data
input_path = Path(args.input)
df = pd.read_csv(input_path)

# Prepare output folder and stem for filenames
output_folder = input_path.parent
input_stem = input_path.stem

# Restrict to given categories
if categories != ["All"]:
    df = df[df["category"].isin(categories)]

# Restrict to given years
if args.years is not None:
    year_span = [i for i in range(min(args.years), max(args.years) + 1)]
    df = df[df["year"].isin(year_span)]

# Display the categories
df_by_category = df.groupby("category", observed=False)
category_counts = df_by_category.size()
category_fig = plt.figure("Category distribution")
plt.pie(category_counts, labels=category_counts.index, autopct="%1.1f%%")
plt.title("Category Distribution")
category_fig.savefig(output_folder / f"{input_stem}_category_distribution.png", dpi=dpi)
plt.show()

# Display the subcategories
df_by_subcategory = df.groupby("subcategory", observed=False)
subcategory_counts = df_by_subcategory.size()
subcategory_fig = plt.figure("Subcategory distribution")
plt.pie(subcategory_counts, labels=subcategory_counts.index, autopct="%1.1f%%")
plt.title("Subcategory Distribution")
subcategory_fig.savefig(
    output_folder / f"{input_stem}_subcategory_distribution.png", dpi=dpi
)
plt.show()

# --- Subcategory count per year: line plot ---
subcategory_palette = plt.get_cmap("tab20")
subcategory_colors = {
    subcat: subcategory_palette(i % 20)
    for i, subcat in enumerate(subcategory_counts.index)
}
paper_colors = {
    "Not open": "tab:red",
    "Open access": "tab:green",
}
data_colors = {
    "Not open": "tab:red",
    "On request": "#FFC300",  # bright dark yellow
    "Open access": "tab:green",
}

subcategory_year_counts = (
    df.groupby(["year", "subcategory"], observed=False).size().unstack(fill_value=0)
)
fig_subcat_line, ax = plt.subplots(figsize=(8, 5))
for i, subcat in enumerate(subcategory_counts.index):
    ax.plot(
        subcategory_year_counts.index,
        subcategory_year_counts[subcat],
        label=subcat,
        color=subcategory_colors[subcat],
        marker="o",
    )
ax.set_xlabel("Year")
ax.set_ylabel("Number of submissions")
ax.set_title("Number of Submissions per Subcategory per Year")
ax.legend(title="Subcategory", bbox_to_anchor=(1.05, 1), loc="upper left")
fig_subcat_line.tight_layout()
fig_subcat_line.savefig(
    output_folder / f"{input_stem}_subcategory_per_year.png", dpi=dpi
)
plt.show()

# --- Category count per year: line plot ---
category_year_counts = (
    df.groupby(["year", "category"], observed=False).size().unstack(fill_value=0)
)
fig_cat_line, ax = plt.subplots(figsize=(8, 5))
for i, cat in enumerate(category_counts.index):
    ax.plot(
        category_year_counts.index,
        category_year_counts[cat],
        label=cat,
        marker="o",
    )
ax.set_xlabel("Year")
ax.set_ylabel("Number of submissions")
ax.set_title("Number of Submissions per Category per Year")
ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
fig_cat_line.tight_layout()
fig_cat_line.savefig(output_folder / f"{input_stem}_category_per_year.png", dpi=dpi)
plt.show()

# --- Grouped Pie Charts by 5-year Periods: Paper and Data Availability ---
# Create 5-year bins
admissible_starts = set([2021 - 5 * i for i in range(0, 20)])
if "year" in df.columns:
    bin_size = 5
    min_year = int(df["year"].min())
    start = max([s for s in admissible_starts if s <= min_year])
    end = int(df["year"].max()) + 1
    bins = list(range(start, end + 1, bin_size))
    labels = [f"{bins[i]}-{bins[i + 1] - 1}" for i in range(len(bins) - 1)]
    labels[0] = f"{max(bins[0], min_year)}-{bins[1] - 1}"
    df["Period"] = pd.cut(df["year"], bins=bins, labels=labels, right=False)

    # Standardize paper/data availability columns if present
    paper_col = "article_availability"
    data_col = "data_availability"

    # Map scores 0 to "Not open", 0.5 to "On request", 1 to "Open access"
    paper_availability_map = {0: "Not open", 1: "Open access"}
    data_availability_map = {0: "Not open", 0.5: "On request", 1: "Open access"}
    df[paper_col] = df[paper_col + "_score"].map(paper_availability_map)
    df[data_col] = df[data_col + "_score"].map(data_availability_map)

    # Paper and data availability pie charts by period, per category
    if paper_col and data_col and "category" in df.columns:
        for cat in categories:
            df_cat = df[df["category"] == cat]
            if df_cat.empty:
                continue
            # Paper availability
            paper_order = ["Open access", "Not open"]
            trend_paper = (
                df_cat.groupby(["Period", paper_col], observed=False)
                .size()
                .unstack(fill_value=0)
            )
            trend_paper = trend_paper.reindex(columns=paper_order, fill_value=0)
            fig_paper, axes_paper = plt.subplots(
                1, len(trend_paper.index), figsize=(4 * len(trend_paper.index), 4)
            )
            if len(trend_paper.index) == 1:
                axes_paper = [axes_paper]
            for i, period in enumerate(trend_paper.index):
                values = trend_paper.loc[period]
                # Allow to control the radius using args.pie_size
                wedges, texts, autotexts = axes_paper[i].pie(
                    values,
                    labels=None,
                    autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
                    startangle=90,
                    counterclock=False,
                    textprops={"color": "white", "fontsize": 14},
                    wedgeprops={"edgecolor": "white", "linewidth": 1},
                    colors=[paper_colors[c] for c in values.index],
                    radius=args.pie_size,
                )
                for autotext in autotexts:
                    autotext.set_color("white")
                    autotext.set_fontweight("bold")
                axes_paper[i].text(
                    0.5,
                    -0.05,
                    f"{period}",
                    ha="center",
                    va="center",
                    transform=axes_paper[i].transAxes,
                    fontsize=14,
                )
                total = int(values.sum())
                axes_paper[i].text(
                    0.5,
                    -0.15,
                    f"#total: {total}",
                    ha="center",
                    va="center",
                    transform=axes_paper[i].transAxes,
                    fontsize=12,
                )
            axes_paper[-1].legend(
                wedges,
                values.index,
                title="Paper Availability",
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )
            fig_paper.text(
                args.horizontal_shift,
                0.5,
                f"Paper Availability\n{args.journal} | {cat}",
                va="center",
                ha="center",
                rotation=90,
                fontsize=16,
                transform=fig_paper.transFigure,
            )
            plt.tight_layout()
            fig_paper.savefig(
                output_folder / f"{input_stem}_paper_availability_pies_{cat}.png",
                dpi=dpi,
            )
            plt.show()
            # Data availability
            data_order = ["Open access", "On request", "Not open"]
            trend_data = (
                df_cat.groupby(["Period", data_col], observed=False)
                .size()
                .unstack(fill_value=0)
            )
            trend_data = trend_data.reindex(columns=data_order, fill_value=0)
            fig_data, axes_data = plt.subplots(
                1, len(trend_data.index), figsize=(4 * len(trend_data.index), 4)
            )
            if len(trend_data.index) == 1:
                axes_data = [axes_data]
            for i, period in enumerate(trend_data.index):
                values = trend_data.loc[period]
                wedges, texts, autotexts = axes_data[i].pie(
                    values,
                    labels=None,
                    autopct=lambda pct: ""
                    if args.without_labels
                    else f"{pct:.1f}%"
                    if pct > 0
                    else "",
                    startangle=90,
                    counterclock=False,
                    textprops={"color": "white", "fontsize": 14},
                    wedgeprops={"edgecolor": "white", "linewidth": 1},
                    colors=[data_colors[c] for c in values.index],
                    radius=args.pie_size,
                )
                for autotext in autotexts:
                    autotext.set_color("white")
                    autotext.set_fontweight("bold")
                axes_data[i].text(
                    0.5,
                    -0.05,
                    f"{period}",
                    ha="center",
                    va="center",
                    transform=axes_data[i].transAxes,
                    fontsize=14,
                )
                total = int(values.sum())
                axes_data[i].text(
                    0.5,
                    -0.15,
                    f"#total: {total}",
                    ha="center",
                    va="center",
                    transform=axes_data[i].transAxes,
                    fontsize=12,
                )
            axes_data[-1].legend(
                wedges,
                values.index,
                title="Data Availability",
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )
            fig_data.text(
                args.horizontal_shift,
                0.5,
                f"Data Availability\n{args.journal} | {cat}",
                va="center",
                ha="center",
                rotation=90,
                fontsize=16,
                transform=fig_data.transFigure,
            )
            plt.tight_layout()
            fig_data.savefig(
                output_folder / f"{input_stem}_data_availability_pies_{cat}.png",
                dpi=dpi,
            )
            plt.show()

    # --- Paper availability over time (all categories lumped, relative) ---
    if paper_col:
        trend_paper_year = (
            df.groupby(["year", paper_col], observed=False).size().unstack(fill_value=0)
        )
        paper_order = ["Open access", "Not open"]
        trend_paper_year = trend_paper_year.reindex(columns=paper_order, fill_value=0)
        # Compute relative values (percent)
        totals = trend_paper_year.sum(axis=1)
        trend_paper_year_rel = trend_paper_year.divide(totals, axis=0).multiply(100)
        fig, ax = plt.subplots(figsize=(8, 5))
        bottom = np.zeros(len(trend_paper_year_rel))
        for i, label in enumerate(paper_order):
            ax.bar(
                trend_paper_year_rel.index,
                trend_paper_year_rel[label],
                label=label,
                color=paper_colors[label],
                bottom=bottom,
            )
            bottom += trend_paper_year_rel[label].values
        # Add totals on top of bars
        for i, total in enumerate(totals):
            ax.text(
                trend_paper_year_rel.index[i],
                98,
                f"{int(total)}",
                ha="center",
                va="top",
                fontsize=10,
                color="white",
                rotation=90,
            )
        ax.set_xlabel("Year")
        ax.set_ylabel("Percent of articles [%]")
        ax.set_title(f"Paper Availability | {args.journal}")
        ax.legend(
            title="Paper Availability", bbox_to_anchor=(1.05, 1), loc="upper left"
        )
        # Set x-ticks: first, last, and center year
        years = trend_paper_year_rel.index.values
        if len(years) > 2:
            center_idx = len(years) // 2
            xticks = [years[0], years[center_idx], years[-1]]
        else:
            xticks = years
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(y)) for y in xticks])
        fig.tight_layout()
        fig.savefig(
            output_folder / f"{input_stem}_paper_availability_per_year.png", dpi=dpi
        )
        plt.show()

    # --- Data availability over time (all categories lumped, relative) ---
    if data_col:
        trend_data_year = (
            df.groupby(["year", data_col], observed=False).size().unstack(fill_value=0)
        )
        data_order = ["Open access", "On request", "Not open"]
        trend_data_year = trend_data_year.reindex(columns=data_order, fill_value=0)
        # Compute relative values (percent)
        totals = trend_data_year.sum(axis=1)
        trend_data_year_rel = trend_data_year.divide(totals, axis=0).multiply(100)
        fig, ax = plt.subplots(figsize=(8, 5))
        bottom = np.zeros(len(trend_data_year_rel))
        for i, label in enumerate(data_order):
            ax.bar(
                trend_data_year_rel.index,
                trend_data_year_rel[label],
                label=label,
                color=data_colors[label],
                bottom=bottom,
            )
            bottom += trend_data_year_rel[label].values
        # Add totals on top of bars
        for i, total in enumerate(totals):
            ax.text(
                trend_data_year_rel.index[i],
                98,
                f"{int(total)}",
                ha="center",
                va="top",
                fontsize=10,
                color="white",
                rotation=90,
            )
        ax.set_xlabel("Year")
        ax.set_ylabel("Percent of articles [%]")
        ax.set_title(f"Data Availability | {args.journal}")
        ax.legend(title="Data Availability", bbox_to_anchor=(1.05, 1), loc="upper left")
        # Set x-ticks: first, last, and center year
        years = trend_data_year_rel.index.values
        if len(years) > 2:
            center_idx = len(years) // 2
            xticks = [years[0], years[center_idx], years[-1]]
        else:
            xticks = years
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(y)) for y in xticks])
        fig.tight_layout()
        fig.savefig(
            output_folder / f"{input_stem}_data_availability_per_year.png", dpi=dpi
        )
        plt.show()


def statistics_over_time(df, column, entries):
    num_entries = len(entries)
    # Use intuitive colors for open, conditional, closed
    # Open: green, Conditional: orange, Closed: red

    # assert set(df[column].unique().tolist()) == set(entries)
    years = []
    total_counts = {group: [] for group in df[column].unique()}
    open_access_counts = {group: [] for group in df[column].unique()}
    closed_access_counts = {group: [] for group in df[column].unique()}
    open_access_data_availability_counts = {group: [] for group in df[column].unique()}
    closed_access_data_availability_counts = {
        group: [] for group in df[column].unique()
    }
    conditional_access_data_availability_counts = {
        group: [] for group in df[column].unique()
    }

    # Make statistics over time
    df_by_year = df.groupby("year", observed=False)
    for year_value, year_group in df_by_year:
        if year_value <= 2018:
            continue

        years.append(year_value)
        df_by_column = year_group.groupby(column, observed=False)
        for entry in entries:
            if entry not in df_by_column.groups:
                # No entries for this category in this year
                total_counts[entry].append(0)
                open_access_counts[entry].append(0)
                closed_access_counts[entry].append(0)
                open_access_data_availability_counts[entry].append(0)
                closed_access_data_availability_counts[entry].append(0)
                conditional_access_data_availability_counts[entry].append(0)
            else:
                entry_group = df_by_column.get_group(entry)
                total_counts[entry].append(len(entry_group))
                open_access_counts[entry].append(
                    np.sum(entry_group["article_availability_score"] == 1)
                )
                closed_access_counts[entry].append(
                    np.sum(entry_group["article_availability_score"] == 0)
                )
                open_access_data_availability_counts[entry].append(
                    np.sum(entry_group["data_availability_score"] == 1)
                )
                closed_access_data_availability_counts[entry].append(
                    np.sum(entry_group["data_availability_score"] == 0)
                )
                conditional_access_data_availability_counts[entry].append(
                    np.sum(entry_group["data_availability_score"] == 0.5)
                )

    width = 0.6
    x = np.arange(len(years))
    num_entries = len(entries)
    displacement = np.linspace(
        -width / num_entries, width / num_entries, num_entries + 1
    )[:-1]

    # Only keep relative plots (remove absolute/total count plots)
    # --- Relative Article availability over time (stacked bar, %) ---
    fig1 = plt.figure("Relative Article availability over time (stacked bar, %)")
    for idx, entry in enumerate(entries):
        if entry not in total_counts:
            continue
        article_total = np.array(total_counts[entry])
        article_open = np.array(open_access_counts[entry])
        article_closed = np.array(closed_access_counts[entry])
        article_open_rel = np.where(
            article_total > 0, 100 * article_open / article_total, 0
        )
        article_closed_rel = np.where(
            article_total > 0, 100 * article_closed / article_total, 0
        )
        plt.bar(
            x + displacement[idx],
            article_open_rel,
            width=width / num_entries,
            label="Open access",
            color=paper_colors["Open access"],
        )
        plt.bar(
            x + displacement[idx],
            article_closed_rel,
            width=width / num_entries,
            bottom=article_open_rel,
            label="Not open",
            color=paper_colors["Not open"],
        )

    plt.xticks(x, years)
    plt.xlabel("Year")
    plt.ylabel("Percent of articles [%]")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.title("Article Availability Over Time (Stacked Bar, %)")
    plt.ylim(0, 100)
    plt.tight_layout()
    fig1.savefig(output_folder / f"{input_stem}_paper_availability.png", dpi=dpi)
    plt.show()

    # --- Relative Data availability over time (stacked bar, %) ---
    fig2 = plt.figure("Data availability over time (stacked bar, %)")
    for idx, entry in enumerate(entries):
        if entry not in total_counts:
            continue
        data_total = np.array(total_counts[entry])
        data_open = np.array(open_access_data_availability_counts[entry])
        data_conditional = np.array(conditional_access_data_availability_counts[entry])
        data_closed = np.array(closed_access_data_availability_counts[entry])
        data_open_rel = np.where(data_total > 0, 100 * data_open / data_total, 0)
        data_conditional_rel = np.where(
            data_total > 0, 100 * data_conditional / data_total, 0
        )
        data_closed_rel = np.where(data_total > 0, 100 * data_closed / data_total, 0)
        plt.bar(
            x + displacement[idx],
            data_open_rel,
            width=width / num_entries,
            label="Open access",
            color=data_colors["Open access"],
        )
        plt.bar(
            x + displacement[idx],
            data_conditional_rel,
            width=width / num_entries,
            bottom=data_open_rel,
            label="On request",
            color=data_colors["On request"],
        )
        plt.bar(
            x + displacement[idx],
            data_closed_rel,
            width=width / num_entries,
            bottom=data_open_rel + data_conditional_rel,
            label="Not open",
            color=data_colors["Not open"],
        )

    plt.xticks(x, years)
    plt.xlabel("Year")
    plt.ylabel("Percent of articles [%]")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.title(f"Data Availability | {args.journal}")
    plt.ylim(0, 100)
    plt.tight_layout()
    fig2.savefig(output_folder / f"{input_stem}_data_availability.png", dpi=dpi)
    plt.show()


statistics_over_time(df, "category", categories)
