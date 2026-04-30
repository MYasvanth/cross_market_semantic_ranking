"""Cross-encoder label distillation for synthetic data."""
import logging
import pandas as pd
from sentence_transformers import CrossEncoder

log = logging.getLogger(__name__)


class LabelDistiller:
    """
    Uses a cross-encoder to re-score synthetic query-product pairs
    and map them to a 0-3 relevance scale.

    Only processes rows where pid starts with 'synth_' (synthetic data).
    Leaves ESCI rows untouched.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> CrossEncoder:
        """Lazily load the cross-encoder model."""
        if self._model is None:
            log.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def distill(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Re-score synthetic rows using cross-encoder and map to 0-3 scale.

        Args:
            df: DataFrame with columns [qid, pid, query, product_title, relevance, ...]

        Returns:
            DataFrame with updated relevance for synthetic rows.
        """
        df = df.copy()

        # Identify synthetic rows (pid starts with 'synth_')
        synth_mask = df["pid"].str.startswith("synth_")
        synth_df = df[synth_mask]

        if synth_df.empty:
            log.info("No synthetic rows found, skipping distillation")
            return df

        log.info(f"Distilling {len(synth_df)} synthetic rows using cross-encoder")

        # Run cross-encoder predictions
        query_title_pairs = list(zip(synth_df["query"], synth_df["product_title"]))
        scores = self.model.predict(query_title_pairs, batch_size=256)

        # Map scores from [0,1] to [0,3] scale
        # scores are already in [0,1] from cross-encoder
        new_relevance = (scores * 3).clip(0, 3).round().astype(int)

        # Update only synthetic rows
        df.loc[synth_mask, "relevance"] = new_relevance

        old_dist = synth_df["relevance"].value_counts().sort_index().to_dict()
        new_dist = df.loc[synth_mask, "relevance"].value_counts().sort_index().to_dict()
        log.info(f"Synthetic relevance dist - before: {old_dist}, after: {new_dist}")

        return df
