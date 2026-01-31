import argparse
import logging
import re

from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

# logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DataAvailabilityBot/1.0)"}


def read_keywords_from_csv(filename):
    # Read a single line of comma-separated keywords, strip whitespace
    with open(filename, encoding="utf-8") as f:
        line = f.readline()
        # Use raw string for each keyword to preserve escapes like \b
        return [rf"{key.strip()}" for key in line.strip().split(",") if key.strip()]


def fetch_url(url, timeout=20):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        logging.warning("Request failed for %s: %s", url, e)
        return None


def extract_open_access(soup):
    # Look for <span class="c-article-meta-recommendations__access-type">
    tag = soup.find("span", class_="c-article-meta-recommendations__access-type")
    if tag:
        return tag.get_text(strip=True).lower()
    else:
        "unknown"


def extract_abstract(soup):
    # Search among titles
    _abstract = extract_section(soup, title=["Abstract"])
    if _abstract:
        return _abstract
    # Search in meta tags if not found
    _abstract = extract_meta(soup, content=["Abstract"])
    if _abstract:
        return _abstract
    # Search on unsupported elements
    _abstract = extract_unsupported_element(soup, keywords=["Abstract"])
    # Allow for typos - Search among titles
    _abstract = extract_section(soup, title=["Abstarct"])
    if _abstract:
        return _abstract
    # Allow for typos - Search in meta tags if not found
    _abstract = extract_meta(soup, content=["Abstarct"])
    if _abstract:
        return _abstract


def extract_unsupported_element(soup, keywords):
    tags = soup.find("span", class_="unsupported-element u-hide")
    if not tags:
        return None
    for tag in tags:
        if all(
            re.search(rf"{key}", tag.get_text(strip=True), re.I) for key in keywords
        ):
            return tag.get_text(strip=True).lower()
    return None


def read_dc_type(soup):
    # Find meta with name="dc.type"
    meta = soup.find("meta", attrs={"name": "dc.type"})
    if meta and "content" in meta.attrs:
        return meta["content"]
    return None


def identify_article_type(soup):
    type = read_dc_type(soup)
    for key in ["article", "paper", "report", "letter"]:
        if type and re.search(rf"{key}", type, re.I):
            return "article"
    for key in ["editorial", "acknowledgment"]:
        if type and re.search(rf"{key}", type, re.I):
            return "editorial"
    for key in ["erratum", "correction"]:
        if type and re.search(rf"{key}", type, re.I):
            return "correction"
    for key in ["briefcommunication"]:
        if type and re.search(rf"{key}", type, re.I):
            return "short"
    raise ValueError(f"Unknown article type: {type}")


def count_keywords(text, df):
    if not text:
        return
    counts = []
    for _, row in df.iterrows():
        key = row["keyword"]
        count = len(re.findall(rf"{key}", text, re.I))
        counts.append(count)
    df["keyword_counter"] = counts
    return


def find_largest_counter(df_list, column):
    total_counts = [sum(df[column]) for df in df_list]
    if sum(total_counts) == 0:
        return None
    max_index = total_counts.index(max(total_counts))
    return max_index


def extract_matching_keywords(df_list, column_key, column_count):
    keywords = [
        row[column_key]
        for df in df_list
        for _, row in df.iterrows()
        if row[column_count] > 0
    ]
    return ", ".join(keywords)


def count_category(df):
    for column in ["category", "subcategory", "subcategory2"]:
        # Make unique and remove NaN categories
        categories = df[column].unique()
        categories = [cat for cat in categories if pd.notna(cat)]
        counter = {
            category: sum(
                [
                    row["keyword_counter"]
                    for _, row in df.iterrows()
                    if row[column] == category
                ]
            )
            for category in categories
        }
        category_counter = []
        for _, row in df.iterrows():
            if pd.isna(row[column]):
                category_counter.append(0)
            else:
                category_counter.append(counter[row[column]])
        df[f"{column}_counter"] = category_counter
    return df


def find_frequent_key(df, column_key, column_count):
    # Find the index of the max value in the column_count column
    max_index = df[column_count].idxmax()
    max_value = df.loc[max_index, column_key]
    return max_value


def extract_section(soup, title: list):
    sections = soup.find_all("section", attrs={"data-title": True})
    sections_list = [sec for sec in sections]
    section_titles = [
        sec["data-title"] for sec in sections if "data-title" in sec.attrs
    ]
    # Count matches for each title keyword
    title_matches = [
        sum([1 for key in title if re.search(rf"{key}", sec_title, re.I)])
        for sec_title in section_titles
    ]
    if not title_matches or max(title_matches) < len(title):
        return None

    # Extract the section with the highest match count
    max_index = title_matches.index(max(title_matches))
    section = sections_list[max_index]
    return section.text


def extract_meta(soup, content: list):
    # Check <meta> tags for keywords in content
    metas = soup.find_all("meta")
    metas_list = [meta for meta in metas]
    meta_matches = [
        sum(
            [
                1
                for key in content
                if re.search(rf"{key}", meta.get("content", ""), re.I)
            ]
        )
        for meta in metas_list
    ]
    # Found nothing
    if not meta_matches or max(meta_matches) == 0:
        return None
    # Extract the meta with the highest match count
    max_index = meta_matches.index(max(meta_matches))
    meta = metas_list[max_index]
    return meta.get("content")


def score(text, evaluation_df, empty_category="none"):
    if not text:
        return 0, empty_category
    scores = []
    categories = evaluation_df.get("category", [])
    for _, row in evaluation_df.iterrows():
        key = row["keyword"]
        score = row["score"]
        if re.search(rf"{key}", text, re.I):
            scores.append(score)
        else:
            scores.append(0)
    if not scores or all(s == 0 for s in scores):
        return 0, empty_category
    return max(scores), categories[scores.index(max(scores))]


def save_soup_to_file(soup, filename="soup.html"):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(str(soup))


def determine_keywords(df):
    # Extract active keywords
    active_df = df[df["keyword_counter"] > 0].dropna(subset=["keyword"])
    return ", ".join(active_df["keyword"].tolist())


def determine_classification(df):
    # Extract active classifications
    classification = []
    active_df = df[df["category_counter"] > 0].dropna(subset=["category"])
    classification = [row["category"] for _, row in active_df.iterrows()]
    active_df = active_df[active_df["subcategory_counter"] > 0].dropna(
        subset=["subcategory"]
    )
    classification += [row["subcategory"] for _, row in active_df.iterrows()]
    active_df = active_df[active_df["subcategory2_counter"] > 0].dropna(
        subset=["subcategory2"]
    )
    classification += [row["subcategory2"] for _, row in active_df.iterrows()]
    return ", ".join(set(classification))


def determine_category(df):
    # Extract active categories
    active_df = df[df["category_counter"] > 0].dropna(subset=["category"])
    active_categories = set(
        [(row["category"], row["category_counter"]) for _, row in active_df.iterrows()]
    )
    if len(active_categories) == 0:
        return ("other",)
    else:
        priority = {"imaging": 0, "simulation": 1, "other": 2}
        sorted_categories = sorted(
            active_categories, key=lambda x: (x[1], priority.get(x[0], 0)), reverse=True
        )
        # In case of a tie, down prioritize "other"
        if (
            len(sorted_categories) > 1
            and sorted_categories[0][1] == sorted_categories[1][1]
            and sorted_categories[0][0] == "other"
        ):
            category = sorted_categories[1][0]
        else:
            category = sorted_categories[0][0]

    # Extract active subcategories
    active_df = active_df.groupby("category").get_group(category)
    active_df = active_df[active_df["subcategory_counter"] > 0].dropna(
        subset=["subcategory"]
    )
    active_subcategories = [
        (row["subcategory"], row["subcategory_counter"])
        for _, row in active_df.iterrows()
    ]
    if len(active_subcategories) == 0:
        return (category,)
    else:
        subcategory = sorted(
            set(active_subcategories), key=lambda x: x[1], reverse=True
        )[0][0]

    # Extract active subcategories2
    active_df = active_df[active_df["subcategory"] == subcategory].dropna(
        subset=["subcategory2"]
    )
    active_df = active_df[active_df["subcategory2_counter"] > 0]
    active_subcategories2 = [
        (row["subcategory2"], row["subcategory2_counter"])
        for _, row in active_df.iterrows()
    ]
    if len(active_subcategories2) == 0:
        return (category, subcategory)
    else:
        subcategory2 = sorted(
            set(active_subcategories2), key=lambda x: x[1], reverse=True
        )[0][0]

    return (category, subcategory, subcategory2)


def main(
    input_csv,
    output_csv,
    categories_csv,
    open_access_availablity_csv,
    data_availability_csv,
):
    # Read keywords from CSV files as DataFrames
    df = pd.read_csv(input_csv, dtype=str)
    data_availability_scores_df = pd.read_csv(data_availability_csv)
    open_access_scores_df = pd.read_csv(open_access_availablity_csv)
    categories_df = pd.read_csv(categories_csv)

    # Count articles
    num_redefined = 0
    logging.info("Read %d rows from %s", len(df), input_csv)

    # Unify the categories from TiPM style to generic style
    df["year"] = df["Publication Year"]
    df["title"] = df["Item Title"]
    df["doi"] = df["Item DOI"]
    df["type"] = (
        df["Content Type"]
        .map(
            {
                "Article": "article",
                "Review": "article",
                "Book Chapter": "book-chapter",
                "Book": "book",
                "Conference Paper": "conference-paper",
                "Data Paper": "data-paper",
                "Editorial": "editorial",
                "Letter": "letter",
                "News": "news",
                "Correction": "correction",
                "Retraction": "retraction",
                # Add more mappings as needed
            }
        )
        .fillna("other")
    )
    df["url"] = df["URL"]
    df["journal"] = df["Publication Title"]
    df.drop(
        columns=[
            "Publication Year",  # -> "year"
            "Item Title",  # -> "title"
            "Item DOI",  # -> "doi"
            "Book Series Title",  # [nan]
            "Publication Title",  # -> "journal"
            "Journal Volume",  # [nan]
            "Journal Issue",  # [nan]
            "Item DOI",  # -> doi
            "URL",  # -> url
            "Content Type",  # -> type
        ],
        errors="ignore",
        inplace=True,
    )

    # Stop if type is "other"
    if (df["type"] == "other").any():
        raise ValueError("Unsupported article type found")

    # Containers for results
    rights_and_permission_score = []
    rights_and_permission_category = []
    rights_and_permission_section = []
    category = []
    subcategory = []
    subcategory2 = []
    keywords = []
    classification = []
    data_availability_score = []
    data_availability_category = []
    abstract = []
    data_availability_section = []

    for idx in range(len(df)):
        url = df["url"].iloc[idx]
        path = Path(f"soups_{input_csv.stem}/soup_{idx}.html")
        logging.info("[%d] Fetching %s", idx, url)
        if Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
        else:
            # Fetch url
            r = fetch_url(url)
            if r is None:
                raise ValueError("Failed to fetch URL: %s", url)

            # Convert to soup
            soup = BeautifulSoup(r.text, "html.parser")
            save_soup_to_file(soup, filename=path)

        # Identify article type - only continue for "article"
        _article_type = identify_article_type(soup)
        if _article_type != "article":
            category.append(_article_type)
            subcategory.append("N/A")
            subcategory2.append("N/A")
            abstract.append("N/A")
            keywords.append("N/A")
            classification.append("N/A")
            rights_and_permission_section.append("N/A")
            rights_and_permission_score.append(0)
            rights_and_permission_category.append("N/A")
            data_availability_section.append("N/A")
            data_availability_score.append(0)
            data_availability_category.append("N/A")
            continue

        # Extract abstract - required
        _abstract = extract_abstract(soup)
        if not _abstract:
            category.append("error")
            subcategory.append("N/A")
            subcategory2.append("N/A")
            abstract.append("N/A")
            keywords.append("N/A")
            classification.append("N/A")
            rights_and_permission_section.append("N/A")
            rights_and_permission_score.append(0)
            rights_and_permission_category.append("N/A")
            data_availability_section.append("N/A")
            data_availability_score.append(0)
            data_availability_category.append("N/A")
            continue
        abstract.append(_abstract)

        # Extract rights and permissions for article - required
        rights_and_permissions_section = extract_section(
            soup, title=["rights", "permission"]
        )
        if not rights_and_permissions_section:
            raise ValueError(f"Rights and permissions section not found for {url}")
        rights_and_permission_section.append(rights_and_permissions_section)

        # Score rights and permission
        _rights_and_permission_score, _rights_and_permission_category = score(
            rights_and_permissions_section,
            open_access_scores_df,
            empty_category="closed access",
        )
        rights_and_permission_score.append(_rights_and_permission_score)
        rights_and_permission_category.append(_rights_and_permission_category)

        # Initialize article-specific df for keyword counting
        _df = categories_df.copy()

        # Find keywords
        count_keywords(_abstract, _df)
        _keywords = determine_keywords(_df)
        keywords.append(_keywords)

        # Find category
        _df = count_category(_df)
        _category = determine_category(_df)
        while len(_category) < 3:
            _category = _category + ("N/A",)
        category.append(_category[0])
        subcategory.append(_category[1])
        subcategory2.append(_category[2])

        # Find classification
        _classification = determine_classification(_df)
        classification.append(_classification)

        # Extract section on data/code availability
        def none_to_txt(x):
            return "" if x is None else x

        _data_availability_section = ""
        _data_availability_section += none_to_txt(
            extract_section(soup, title=["data", "avail"])
        )
        _data_availability_section += none_to_txt(
            extract_section(soup, title=["code", "avail"])
        )
        # Some articles use "Notes" or the "Acknowledgements" section - only add if nothing else found
        _data_availability_section += none_to_txt(
            extract_section(soup, title=["notes"])
        )
        _data_availability_section += none_to_txt(
            extract_section(soup, title=["acknowledgements"])
        )
        # some articles use the Ethics section
        _data_availability_section += none_to_txt(
            extract_section(soup, title=["ethics"])
        )
        # some articles use the additional information
        _data_availability_section += none_to_txt(
            extract_section(soup, title=["additional", "information"])
        )
        data_availability_section.append(_data_availability_section)

        # Score data availability
        _data_availability_score, _data_availability_category = score(
            _data_availability_section,
            data_availability_scores_df,
            empty_category="closed access",
        )
        data_availability_score.append(_data_availability_score)
        data_availability_category.append(_data_availability_category)

    # Inform on redefinitions
    if num_redefined > 0:
        logging.info(
            "Reclassified %d theoretical articles to computational due to full data availability score of 1.0",
            num_redefined,
        )

    # Update data frame
    df["category"] = category
    df["subcategory"] = subcategory
    df["subcategory2"] = subcategory2
    df["abstract"] = abstract
    df["keywords"] = keywords
    df["classification"] = classification
    df["article_availability_score"] = rights_and_permission_score
    df["article_availability_category"] = rights_and_permission_category
    df["article_availability_section"] = rights_and_permission_section
    df["data_availability_score"] = data_availability_score
    df["data_availability_category"] = data_availability_category
    df["data_availability_section"] = data_availability_section

    # Store data frame to file
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logging.info("Wrote results to %s", output_csv)

    # Count the final categories
    df_by_category = df.groupby("category", observed=False)
    category_counts = df_by_category.size()
    print("Final category counts:")
    print(category_counts)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Extract 'Data availability' text from article URLs in a CSV"
    )
    p.add_argument("--input", "-i", required=True, help="input CSV file")
    p.add_argument("--output", "-o", required=True, help="output CSV file")
    p.add_argument("--categories", "-c", required=True, help="categories CSV file")
    p.add_argument(
        "--open_access", "-oa", required=True, help="open access availability CSV file"
    )
    p.add_argument(
        "--data_availability", "-da", required=True, help="data availability CSV file"
    )
    args = p.parse_args()
    main(
        Path(args.input),
        Path(args.output),
        Path(args.categories),
        Path(args.open_access),
        Path(args.data_availability),
    )
