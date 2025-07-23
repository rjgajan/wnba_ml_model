# wnba_ml_model

This project aims to predict individual player points scored in WNBA games using machine learning models. By leveraging historical box scores, player statistics, team performance data, and contextual features (e.g., opponent defense, pace), the model seeks to provide accurate point projections for individual players in upcoming games.

Current features: average points in L15, average points in L5, average points in L3 vs. opponent, opponent defensive rating (as recorded on espn.com), days of rest
Ongoing edits: change days of rest to rest differential for better capture of how rest will impact offense/defense, incorporate usage rate (likely fg attempted as that is most important usage metric for scoring, can pull fg attempt projections from sportsbook as a per-game input), split model between forwards and guards and use more specific defensive metrics (team might have strong defense in the paint i.e. Chicago Sky but might be weaker on the perimeter)