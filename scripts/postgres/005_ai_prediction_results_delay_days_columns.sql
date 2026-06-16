ALTER TABLE ai_prediction_results
    ADD COLUMN IF NOT EXISTS predicted_delay_days NUMERIC(6, 2);