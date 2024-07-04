import pandas as pd


class DataConvert:
    
    @classmethod
    def convert_csv_to_parquet(cls, csv_path, parquet_path):
        df = pd.read_csv(csv_path)
        df.to_parquet(parquet_path)
        print(f"Converted {csv_path} to {parquet_path}")
        return
    
