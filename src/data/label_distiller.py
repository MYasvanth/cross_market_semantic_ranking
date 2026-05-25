"""Label distillation — cross-encoder teacher scoring for ESCI rows only.

Synthetic rows are never distilled:
  - Queries are template-derived from product titles → bi-encoder similarity
    is artificially high regardless of true relevance.
  - Distilling synthetic rows collapses all labels toward 2-3 and destroys
    the label=0 signal the ranker needs.

ESCI rows benefit from distillation:
  - Human labels are coarse 4-level buckets (Exact/Substitute/Complement/Irrelevant).
  - The cross-encoder produces continuous scores that distinguish strong Exact
    from weak Exact, and borderline Substitute from Complement.
  - Soft continuous labels give LambdaMART finer-grained gradient signal.

GPU requirement:
  - Scoring 100k ESCI pairs at batch_size=256 takes ~8 min on GPU, ~2h on CPU.
  - Set use_cross_encoder_distillation=True only when GPU is available.
"""
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class LabelDistiller:
    """
    Cross-encoder teacher that replaces coarse ESCI ordinal labels with
    continuous soft scores normalized to the [0, 3] relevance scale.

    Only applied to ESCI rows (pid does not start with 'synth_').
    Synthetic labels from assign_relevance() are preserved unchanged.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",  # multilingual, 41 langs
        batch_size: int = 256,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

    def distill(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score ESCI (query, product_title) pairs with the cross-encoder teacher.
        Replaces df['relevance'] for ESCI rows with continuous soft labels.

        Args:
            df: DataFrame with columns [qid, pid, query, product_title, relevance]

        Returns:
            DataFrame with soft labels on ESCI rows, integer labels on synthetic rows.
        """
        synth_mask = df["pid"].str.startswith("synth_")
        n_synth    = synth_mask.sum()
        n_esci     = (~synth_mask).sum()

        if n_esci == 0:
            log.info("No ESCI rows found — distillation skipped, synthetic labels preserved.")
            return df

        log.info(f"Distilling {n_esci} ESCI rows (synthetic {n_synth} rows unchanged).")

        from sentence_transformers import CrossEncoder
        model = CrossEncoder(self.model_name)

        esci_df   = df[~synth_mask].copy()
        queries   = esci_df["query"].tolist()
        titles    = esci_df["product_title"].tolist()
        pairs     = list(zip(queries, titles))

        log.info(f"Scoring {len(pairs)} pairs with {self.model_name} (batch_size={self.batch_size})")
        raw_scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=True)

        # Normalize per-query to [0, 3] — preserves relative ordering within each query group
        # while keeping the same scale as ESCI ordinal labels for LambdaMART
        esci_df["relevance"] = self._normalize_per_query(
            raw_scores, esci_df["qid"].values
        )

        # Log label distribution after distillation
        soft_dist = {
            "mean": float(np.mean(esci_df["relevance"])),
            "std":  float(np.std(esci_df["relevance"])),
            "min":  float(np.min(esci_df["relevance"])),
            "max":  float(np.max(esci_df["relevance"])),
        }
        log.info(f"Soft label distribution (ESCI): {soft_dist}")

        # Recombine: synthetic rows keep integer labels, ESCI rows get soft labels
        df = pd.concat([df[synth_mask], esci_df], ignore_index=True)
        return df

    @staticmethod
    def _normalize_per_query(
        scores: np.ndarray, qids: np.ndarray, scale: float = 3.0
    ) -> np.ndarray:
        """
        Normalize cross-encoder scores to [0, scale] per query group.

        Per-query normalization ensures that within each query, the most
        relevant product gets score=3.0 and the least relevant gets score=0.0,
        regardless of the absolute cross-encoder score range.
        This is critical for LambdaMART which computes gradients from
        within-group relative ordering, not absolute scores.
        """
        normalized = np.zeros_like(scores, dtype=np.float32)
        for qid in np.unique(qids):
            mask      = qids == qid
            q_scores  = scores[mask]
            q_min, q_max = q_scores.min(), q_scores.max()
            if q_max > q_min:
                normalized[mask] = scale * (q_scores - q_min) / (q_max - q_min)
            elif mask.sum() == 1:
                # Single-product group: preserve the original ordinal label
                # rather than assigning an arbitrary midpoint.
                # Caller keeps integer label for this row unchanged.
                normalized[mask] = q_scores[0]
            else:
                # All scores identical in a multi-product group — assign midpoint.
                normalized[mask] = scale / 2.0
        return normalized
