# wnba_ml_model

This project aims to predict individual player points scored in WNBA games using machine learning models. By leveraging historical box scores, player statistics, team performance data, and contextual features (e.g., opponent defense, pace), the model seeks to provide accurate point projections for individual players in upcoming games.

Current features: average points in L15, average points in L5, average points in L3 vs. opponent, opponent defensive rating (as recorded on espn.com), days of rest

Ongoing edits:
- change days of rest to rest differential for better capture of how rest will impact offense/defense *DONE*
- incorporate usage rate
- split model between forwards and guards and use more specific defensive metrics (team might have strong defense in the paint i.e. Chicago Sky but might be weaker on the perimeter)
- pace (poss/40) *DONE*
- drop L3 vs. OPP (field was empty or only had one or two games too often and increased RMSE by 0.5) *DONE*
- add a measure of predicted or actual game spread (account for blowouts / chance of overtime)