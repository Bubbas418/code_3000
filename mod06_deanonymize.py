import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    keys = ["age", "gender", "zip3"]
    unique_anon = ~anon_df.duplicated(subset=keys, keep=False)
    unique_aux = ~aux_df.duplicated(subset=keys, keep=False)
    valid_anon = anon_df.loc[unique_anon, keys + ["anon_id"]]
    valid_aux = aux_df.loc[unique_aux, keys + ["name"]]
    matches = pd.merge(valid_anon, valid_aux, on=keys, how="inner")
    return matches[["anon_id", "name"]].rename(
        columns={"name": "matched_name"}
    )


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    return len(matches_df) / len(anon_df)
