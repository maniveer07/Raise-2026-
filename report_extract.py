import pandas as pd
import numpy as np
import re
from pathlib import Path

USE_DATASET = "A"
DATA_DIR = Path("data")
NEWS_FILES = {
    "A": "dataset_A_news_full_10500.csv",
    "B": "dataset_B_news_subset_3500.csv",
}
DATASET_C_CANDIDATES = [
    "Dataset_C_prompts_&_queries.csv",
    "Dataset_C_prompts_-_queries.csv",
]


def normalize_colname(col: str) -> str:
    col = col.strip().lower()
    col = re.sub(r"[\s\-]+", "_", col)
    col = re.sub(r"[^a-z0-9_]+", "", col)
    return col


def find_column(norm_map: dict, candidates: set) -> str | None:
    for original, normed in norm_map.items():
        if normed in candidates:
            return original
    return None


def resolve_news_path(use_dataset: str) -> Path:
    if use_dataset not in NEWS_FILES:
        raise ValueError
    path = DATA_DIR / NEWS_FILES[use_dataset]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def resolve_dataset_c_path() -> Path:
    for name in DATASET_C_CANDIDATES:
        path = DATA_DIR / name
        if path.exists():
            return path
    raise FileNotFoundError


def load_news(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    norm_map = {c: normalize_colname(c) for c in df.columns}
    title_col = find_column(norm_map, {"title", "headline", "news_title"})
    source_col = find_column(norm_map, {"source", "publisher", "media_source", "outlet"})
    date_col = find_column(norm_map, {"date", "published", "publish_date", "datetime", "timestamp"})
    classes_col = find_column(
        norm_map,
        {"classesstr", "classes_str", "classes", "class", "labels", "categories", "category"},
    )
    id_col = find_column(norm_map, {"headline_id", "news_id", "article_id", "doc_id", "id"})
    if any(c is None for c in [title_col, source_col, date_col, classes_col]):
        raise ValueError(f"Missing required columns. Have: {list(df.columns)}")
    rename_map = {
        title_col: "title",
        source_col: "source",
        date_col: "date",
        classes_col: "classesstr",
    }
    if id_col is not None:
        rename_map[id_col] = "headline_id"
    df = df.rename(columns=rename_map)
    if "headline_id" not in df.columns:
        df["headline_id"] = np.arange(len(df))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["dayofweek"] = df["date"].dt.dayofweek
    df["isweekend"] = df["dayofweek"] >= 5
    df["title"] = df["title"].astype(str)
    df["source"] = df["source"].astype(str)
    df["classesstr"] = df["classesstr"].fillna("").astype(str)
    return df


def load_dataset_c(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    norm_map = {c: normalize_colname(c) for c in df.columns}
    prompt_col = find_column(norm_map, {"prompt", "instruction", "persona_prompt"})
    query_col = find_column(norm_map, {"query", "question"})
    llm_col = find_column(norm_map, {"llm", "model", "model_name"})
    output_col = find_column(
        norm_map, {"llmoutput", "llm_output", "output", "response", "completion"}
    )
    if any(c is None for c in [prompt_col, query_col, llm_col, output_col]):
        raise ValueError(f"Missing required columns in Dataset C. Have: {list(df.columns)}")
    df = df.rename(
        columns={
            prompt_col: "Prompt",
            query_col: "Query",
            llm_col: "LLM",
            output_col: "LLMoutput",
        }
    )
    return df


MACRO_CLASSES = {"Society, Ethics & Culture", "Work, Jobs & Economy", "Technology Interaction"}
MICRO_CLASSES = {
    "Routine, Lifestyle & Behavior",
    "Emotion, Motivation & Well-being",
    "Health, Safety & Risk",
    "Learning, Knowledge & Education",
    "Social Interaction & Relationships",
    "Human Roles",
    "Cognitive Decision-Making",
    "Creativity, Expression Identity",
}


def map_class_to_level(cls_name: str) -> str:
    in_micro = cls_name in MICRO_CLASSES
    in_macro = cls_name in MACRO_CLASSES
    if in_micro and in_macro:
        return "both"
    if in_micro:
        return "micro"
    if in_macro:
        return "macro"
    return "unknown"


def explode_classes(df: pd.DataFrame) -> pd.DataFrame:
    exploded = df.copy()
    exploded["class_name"] = exploded["classesstr"].str.split(";")
    exploded = exploded.explode("class_name")
    exploded["class_name"] = exploded["class_name"].str.strip()
    exploded = exploded[exploded["class_name"] != ""]
    return exploded


def summarize_headline_level(levels: pd.Series) -> str:
    has_micro = levels.isin(["micro", "both"]).any()
    has_macro = levels.isin(["macro", "both"]).any()
    if has_micro and has_macro:
        return "both"
    if has_micro:
        return "micro"
    if has_macro:
        return "macro"
    return "unknown"


def level_summary(df: pd.DataFrame, level_col: str = "level") -> pd.DataFrame:
    counts = df[level_col].value_counts(dropna=False).reindex(
        ["micro", "macro", "both", "unknown"], fill_value=0
    )
    pct = (counts / counts.sum() * 100).round(1)
    return pd.DataFrame({"count": counts, "pct": pct})


MICRO_KEYWORDS = [
    "daily",
    "day-to-day",
    "routine",
    "habit",
    "schedule",
    "morning",
    "sleep",
    "diet",
    "exercise",
    "productivity",
    "study",
    "homework",
    "class",
    "school",
    "university",
    "family",
    "friends",
    "relationship",
    "partner",
    "mental health",
    "stress",
    "anxiety",
    "motivation",
    "focus",
    "workday",
    "office",
    "coworkers",
    "social media",
    "apps",
]
MACRO_KEYWORDS = [
    "policy",
    "regulation",
    "regulatory",
    "government",
    "governance",
    "law",
    "legal",
    "democracy",
    "society",
    "societal",
    "economy",
    "economic",
    "labor market",
    "job market",
    "inequality",
    "rights",
    "privacy",
    "fairness",
    "bias",
    "security",
    "national security",
    "infrastructure",
    "big tech",
    "platforms",
    "industry-wide",
]


def extract_persona(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        return "Unknown"
    for pattern in [r"Role\s*[:\-]\s*(.+)", r"Persona\s*[:\-]\s*(.+)"]:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip().splitlines()[0].strip()
            return re.sub(r"[\s\|\-]+$", "", value)[:80]
    first_line = prompt.strip().splitlines()[0]
    first_line = re.sub(r"^\s*(You are|Youre|You are an?)\s+", "", first_line, flags=re.IGNORECASE)
    return first_line[:80]


def tag_llm_output_level(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    lower = text.lower()
    has_micro = any(k in lower for k in MICRO_KEYWORDS)
    has_macro = any(k in lower for k in MACRO_KEYWORDS)
    if has_micro and has_macro:
        return "both"
    if has_micro:
        return "micro"
    if has_macro:
        return "macro"
    return "unknown"


def main() -> None:
    news_df = load_news(resolve_news_path(USE_DATASET))
    news_exploded = explode_classes(news_df)
    news_exploded["class_level"] = news_exploded["class_name"].apply(map_class_to_level)
    headline_levels = (
        news_exploded.groupby("headline_id")["class_level"]
        .apply(summarize_headline_level)
        .reset_index()
        .rename(columns={"class_level": "level"})
    )
    news_headline = news_df.merge(headline_levels, on="headline_id", how="left")
    news_headline["level"] = news_headline["level"].fillna("unknown")

    persona_df = load_dataset_c(resolve_dataset_c_path())
    persona_df = persona_df.copy()
    persona_df["persona"] = persona_df["Prompt"].apply(extract_persona)
    combined_text = persona_df["LLMoutput"].fillna("").astype(str) + " \n" + persona_df[
        "Query"
    ].fillna("").astype(str)
    persona_df["level"] = combined_text.apply(tag_llm_output_level)

    news_level = level_summary(news_headline)
    persona_level = level_summary(persona_df)

    print("NEWS_LEVEL")
    print(news_level.to_string())
    print("\nPERSONA_LEVEL")
    print(persona_level.to_string())

    source_counts = news_headline.groupby("source").size().sort_values(ascending=False)
    top_sources = source_counts.head(10).index
    source_level = (
        news_headline[news_headline["source"].isin(top_sources)]
        .groupby(["source", "level"])
        .size()
        .unstack(fill_value=0)
    )
    source_share = source_level.div(source_level.sum(axis=1), axis=0).round(3)
    print("\nTOP_SOURCES_SHARE")
    print(source_share.to_string())

    persona_counts = persona_df["persona"].value_counts().head(10).index
    persona_level_tbl = (
        persona_df[persona_df["persona"].isin(persona_counts)]
        .groupby(["persona", "level"])
        .size()
        .unstack(fill_value=0)
    )
    persona_share = persona_level_tbl.div(persona_level_tbl.sum(axis=1), axis=0).round(3)
    print("\nTOP_PERSONAS_SHARE")
    print(persona_share.to_string())

    llm_level = persona_df.groupby(["LLM", "level"]).size().unstack(fill_value=0)
    llm_share = llm_level.div(llm_level.sum(axis=1), axis=0).round(3)
    print("\nLLM_SHARE")
    print(llm_share.to_string())

    year_counts = (
        news_headline.dropna(subset=["year"])
        .groupby(["year", "level"])
        .size()
        .unstack(fill_value=0)
    )
    year_share = year_counts.div(year_counts.sum(axis=1), axis=0).round(3)
    print("\nYEAR_SHARE")
    print(year_share[["micro", "macro"]].to_string())


if __name__ == "__main__":
    main()
