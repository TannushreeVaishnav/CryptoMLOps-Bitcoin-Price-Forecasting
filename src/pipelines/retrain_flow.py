from prefect import flow
from src.pipelines.daily_update_flow import daily_update
from src.pipelines.train_flow import train

@flow(name="weekly-btc-retrain")
def weekly_retrain():
    # 1. First ensure data is up to date
    daily_update()
    
    # 2. Retrain the model and evaluate against registry
    train()

if __name__ == "__main__":
    # Scheduled: 00:00 UTC every Sunday
    # To run this scheduling server, run `python src/pipelines/retrain_flow.py`
    weekly_retrain.serve(name="weekly-btc-retrain", cron="0 0 * * 0")
