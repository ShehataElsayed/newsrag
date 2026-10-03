"""Experimental few-shot prompting helpers (private lab code).

Builds the prompt for an instruction model from live, retrieved examples and
turns the model's choice scores into probabilities and an abstain decision.
No model is shipped and no text leaves the caller's machine through this
module. The result is a suggestion for an editor, never a verdict.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .core import ValidationError

_LETTERS = 'ABCDEFGH'


def build_fewshot_prompt(claim: str, examples: Sequence[tuple[str, str]], classes: Sequence[str]) -> tuple[str, list[str]]:
    """Return (prompt, letters). Examples are (text, label), shown in the order given."""
    if not 2 <= len(classes) <= len(_LETTERS):
        raise ValidationError('between 2 and 8 classes are supported')
    letters = list(_LETTERS[:len(classes)])
    legend = '\n'.join(f'{letter} = {name}' for letter, name in zip(letters, classes, strict=True))
    lines = ['You label short news claims with the verdict vocabulary of one fact-checking platform.',
             'Use only the claim text and the labelled examples. Answer with a single letter.', '', legend, '']
    for text, label in examples:
        if label not in classes:
            raise ValidationError(f'unknown example label: {label}')
        lines += [f'Claim: {text}', f'Verdict: {letters[list(classes).index(label)]}', '']
    lines += [f'Claim: {claim}', 'Verdict:']
    return '\n'.join(lines), letters


def choice_probabilities(scores: Sequence[float]) -> list[float]:
    """Softmax over the model's scores for the answer letters."""
    if not scores:
        raise ValidationError('scores must not be empty')
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


@dataclass(frozen=True)
class FewShotDecision:
    """Suggestion or abstention from a few-shot answer."""

    suggestion: str | None
    confidence: float
    abstain: bool
    validated_for_release: bool = False
    status: str = 'suggestion for an editor; experimental'


def decide_fewshot(probabilities: Sequence[float], classes: Sequence[str], min_confidence: float) -> FewShotDecision:
    """Answer with the top class when its probability reaches min_confidence, else abstain."""
    if len(probabilities) != len(classes) or not 0.0 < min_confidence <= 1.0:
        raise ValidationError('probabilities must match classes and min_confidence must be in (0, 1]')
    best = max(range(len(classes)), key=lambda i: probabilities[i])
    conf = probabilities[best]
    return FewShotDecision(None if conf < min_confidence else classes[best], conf, conf < min_confidence)
