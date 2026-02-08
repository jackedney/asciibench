"""Tests for the subject_extractor module."""

from asciibench.evaluator.subject_extractor import extract_subject


class TestExtractSubject:
    """Tests for extract_subject function."""

    def test_simple_subject_draw_a_cat_in_ascii_art(self):
        """Extract 'cat' from 'Draw a cat in ASCII art'."""
        result = extract_subject("Draw a cat in ASCII art")
        assert result == "cat"

    def test_spatial_relationship_cat_sitting_on_fence(self):
        """Extract 'cat, fence' from 'Draw a cat sitting on a fence'."""
        result = extract_subject("Draw a cat sitting on a fence")
        assert result == "cat, fence"

    def test_simple_subject_draw_a_house(self):
        """Extract 'house' from 'Draw a house'."""
        result = extract_subject("Draw a house")
        assert result == "house"

    def test_spatial_relationship_bird_above_tree(self):
        """Extract 'bird, tree' from 'Draw a bird above a tree'."""
        result = extract_subject("Draw a bird above a tree")
        assert result == "bird, tree"

    def test_spatial_relationship_dog_under_table(self):
        """Extract 'dog, table' from 'Draw a dog under a table'."""
        result = extract_subject("Draw a dog under a table")
        assert result == "dog, table"

    def test_spatial_relationship_beside(self):
        """Extract both subjects with 'beside' preposition."""
        result = extract_subject("Draw a dog beside a cat")
        assert result == "dog, cat"

    def test_spatial_relationship_next_to(self):
        """Extract both subjects with 'next to' preposition."""
        result = extract_subject("Draw a cat next to a mouse")
        assert result == "cat, mouse"

    def test_spatial_relationship_below(self):
        """Extract both subjects with 'below' preposition."""
        result = extract_subject("Draw a moon below a star")
        assert result == "moon, star"

    def test_spatial_relationship_in_front_of(self):
        """Extract both subjects with 'in front of' preposition."""
        result = extract_subject("Draw a car in front of a building")
        assert result == "car, building"

    def test_spatial_relationship_behind(self):
        """Extract both subjects with 'behind' preposition."""
        result = extract_subject("Draw a tree behind a house")
        assert result == "tree, house"

    def test_subject_with_adjective(self):
        """Extract subject with adjective."""
        result = extract_subject("Draw a happy cat")
        assert result == "happy cat"

    def test_spatial_with_article_an(self):
        """Extract secondary subject with 'an' article."""
        result = extract_subject("Draw a cat on an apple")
        assert result == "cat, apple"

    def test_unrecognized_format(self):
        """Return full prompt text for unrecognized format."""
        result = extract_subject("Unrecognized format")
        assert result == "Unrecognized format"

    def test_empty_string(self):
        """Return empty string for empty input."""
        result = extract_subject("")
        assert result == ""

    def test_whitespace_input(self):
        """Return stripped whitespace for whitespace-only input."""
        result = extract_subject("   ")
        assert result == ""

    def test_draw_without_subject(self):
        """Return full prompt when 'Draw a' has no subject."""
        result = extract_subject("Draw a")
        assert result == "Draw a"

    def test_subject_with_action_no_spatial(self):
        """Extract subject when there's an action but no spatial relationship."""
        result = extract_subject("Draw a cat running fast")
        assert result == "cat"

    def test_case_insensitive(self):
        """Handle case variations."""
        assert extract_subject("DRAW A CAT") == "cat"
        assert extract_subject("draw a CAT") == "cat"
        assert extract_subject("Draw A Cat") == "cat"

    def test_subject_with_trailing_punctuation(self):
        """Handle trailing punctuation in prompt."""
        result = extract_subject("Draw a cat!")
        assert result == "cat"

    def test_spatial_with_multiple_words_in_secondary_subject(self):
        """Extract multi-word secondary subject."""
        result = extract_subject("Draw a cat sitting on a wooden fence")
        assert result == "cat, wooden fence"

    def test_primary_subject_with_trailing_action_words(self):
        """Handle action words after spatial relationship."""
        result = extract_subject("Draw a cat on a fence sleeping")
        assert result == "cat, fence"

    def test_draw_an_article(self):
        """Handle 'Draw an' instead of 'Draw a'."""
        result = extract_subject("Draw an apple")
        assert result == "apple"

    def test_spatial_with_draw_an(self):
        """Handle 'Draw an' with spatial relationship."""
        result = extract_subject("Draw an elephant on a car")
        assert result == "elephant, car"
