from sheets import write_columns_names, export_pandas_df_to_sheets
from dotenv import load_dotenv
import os

if __name__ == "__main__":

    load_dotenv()
    file_path = ""

    if os.getenv("ID") is None:
        spreadsheet_id = write_columns_names(file_path)
        with open(".env", "a") as file:
            file.write(f"ID={spreadsheet_id}")

        export_pandas_df_to_sheets(spreadsheet_id, file_path)
    else:
        spreadsheet_id = os.getenv("ID")
        export_pandas_df_to_sheets(spreadsheet_id, file_path)
        
