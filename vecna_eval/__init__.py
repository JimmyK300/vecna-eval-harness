"""vecna-eval-harness: offline evaluation of multi-provider video retrieval.

Pure standard library. Inputs are saved provider rankings (frame hits with
scores) plus a ground-truth file; outputs are per-arm distinct-video metrics,
per-query first-correct ranks, a gold-conditioned oracle ceiling and paired
bootstrap comparisons.
"""

__version__ = "0.1.0"
