# Cross-Source Record Linking System

This application links records between two datasets that describe the same reality but use different formats and identifiers.

## Features

- **Configurable Rule Tiers**: Set up matching rules with different weights and thresholds
- **Multiple Matching Strategies**: Exact, fuzzy, numeric, and pattern-based matching
- **Transparent Rationale**: Every match includes an explanation of why records were linked
- **Suspect Review**: Review low-confidence matches and adopt new patterns
- **Tie-Breaking**: Resolve multiple matches using configurable tie-breakers
- **Export Functionality**: Download results for downstream use

## How to Use

1. Upload both CSV files (Source A and Source B)
2. Configure matching rules in the sidebar:
   - Activate/deactivate rules
   - Adjust thresholds and weights
   - Set up tie-breakers
3. Click "Run Record Linking" to process the data
4. Review results in the different tabs:
   - Matched: Confidently linked records
   - Suspect: Low-confidence matches that need review
   - Unmatched A/B: Records that couldn't be matched
5. For suspect matches, analyze patterns and adopt new rules if needed
6. Export results using the download button

## Rule Types

- **Exact**: Values must match exactly
- **Fuzzy**: Text similarity using fuzzy matching
- **Numeric**: Numeric values within a tolerance percentage
- **Pattern**: Pattern-based matching (e.g., extract numbers from IDs)

## Installation

1. Ensure Python 3.7+ is installed
2. Install requirements: `pip install streamlit pandas thefuzz`
3. Run the app: `streamlit run app.py`
