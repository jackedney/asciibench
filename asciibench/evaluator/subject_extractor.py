"""Subject extraction from prompt text for VLM evaluation.

This module extracts expected subjects from prompt templates to compare with VLM responses.
It handles various prompt patterns including single subjects and spatial relationships.
"""

import re


def extract_subject(prompt_text: str) -> str:
    """Extract the expected subject(s) from prompt text.

    Args:
        prompt_text: The prompt text to extract subjects from

    Returns:
        The extracted subject(s) as a comma-separated string.
        For single-subject prompts, returns just the subject.
        For spatial relationship prompts, returns both subjects comma-separated.
        If the prompt format is unrecognized, returns the full prompt text.

    Examples:
        >>> extract_subject('Draw a cat in ASCII art')
        'cat'
        >>> extract_subject('Draw a cat sitting on a fence')
        'cat, fence'
        >>> extract_subject('Draw a bird above a tree')
        'bird, tree'
        >>> extract_subject('Draw a dog under a table')
        'dog, table'
        >>> extract_subject('Draw a house')
        'house'
        >>> extract_subject('Unrecognized format')
        'Unrecognized format'
    """
    if not prompt_text:
        return prompt_text

    prompt_text = prompt_text.strip()

    base_pattern = r"Draw\s+(?:a|an)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)"

    base_match = re.search(base_pattern, prompt_text, re.IGNORECASE)

    if not base_match:
        return prompt_text

    primary_subject_words = base_match.group(1).strip().split()

    spatial_preposition_words = {
        "sitting",
        "on",
        "in",
        "under",
        "above",
        "below",
        "beside",
        "next",
        "behind",
        "front",
    }

    action_verbs = {
        "running",
        "flying",
        "swimming",
        "jumping",
        "walking",
        "sleeping",
        "eating",
        "playing",
        "dancing",
        "singing",
        "standing",
    }

    primary_subject = primary_subject_words[0].lower()

    if len(primary_subject_words) > 1:
        second_word = primary_subject_words[1].lower()
        if (
            second_word not in spatial_preposition_words
            and second_word not in action_verbs
            and second_word not in ["a", "an", "the"]
        ):
            primary_subject = f"{primary_subject} {second_word}"

    spatial_prepositions = [
        ("sitting on", 2),
        ("on", 1),
        ("in", 1),
        ("under", 1),
        ("above", 1),
        ("below", 1),
        ("beside", 1),
        ("next to", 2),
        ("behind", 1),
        ("in front of", 3),
    ]

    for prep, _word_count in sorted(spatial_prepositions, key=lambda x: x[1], reverse=True):
        remaining_text = prompt_text[base_match.start() :]
        pattern = (
            rf"{re.escape(primary_subject)}\s+{re.escape(prep)}\s+(?:a|an)\s+"
            rf"([a-zA-Z]+(?:\s+[a-zA-Z]+)*)"
        )
        match = re.search(pattern, remaining_text, re.IGNORECASE)

        if match:
            secondary_subject_words = match.group(1).strip().split()
            secondary_subject = secondary_subject_words[0].lower()
            if len(secondary_subject_words) > 1:
                third_word = secondary_subject_words[1].lower()
                if (
                    third_word not in spatial_preposition_words
                    and third_word not in action_verbs
                    and third_word not in ["a", "an", "the"]
                ):
                    secondary_subject = f"{secondary_subject} {third_word}"
            return f"{primary_subject}, {secondary_subject}"

    return primary_subject
