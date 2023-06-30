from __future__ import print_function
from auth import spreadsheet_service, drive_service
import pandas as pd
import os
from dotenv import load_dotenv


def create_spreadsheet():
    spreadsheet_details = {"properties": {"title": "Attention-Data"}}
    sheet = (
        spreadsheet_service.spreadsheets()
        .create(body=spreadsheet_details, fields="spreadsheetId")
        .execute()
    )

    load_dotenv()
    email = os.getenv("EMAIL_ADDRES")

    spreadsheet_id = sheet.get("spreadsheetId")
    print("Spreadsheet ID: {0}".format(spreadsheet_id))
    permission1 = {
        "type": "user",
        "role": "writer",
        "emailAddress": email,
    }
    drive_service.permissions().create(
        fileId=spreadsheet_id, body=permission1
    ).execute()

    return spreadsheet_id


def create_sheet(spreadsheet_id, title):
    body = {"requests": [{"addSheet": {"properties": {"title": title}}}]}

    result = (
        spreadsheet_service.spreadsheets()
        .batchUpdate(
            spreadsheetId=spreadsheet_id,
            body=body
        ).execute()
    )

    print(f"Sheet {result.get('replies')[0].get('addSheet').get('properties').get('title')} created successfully")


def write_columns_names(spreadsheet_id, file_path, sheet_name="Data"):
    df = pd.read_csv(file_path)
    values = df.columns.to_list()

    range_name = f"{sheet_name}!A1:{chr(ord('A') + len(values) - 1)}1"
    value_input_option = "USER_ENTERED"
    body = {"values": [values]}
    result = (
        spreadsheet_service.spreadsheets()
        .values()
        .update(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption=value_input_option,
            body=body,
        )
        .execute()
    )
    print("{0} cells updated.".format(result.get("updatedCells")))


def write_mean(spreadsheet_id, mean, sheet_name="Mean"):
    range_name = f"{sheet_name}!A1:B1"
    values = [[str(mean)]]
    value_input_option = "USER_ENTERED"
    body = {"values": values}
    result = (
        spreadsheet_service.spreadsheets()
        .values()
        .update(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption=value_input_option,
            body=body,
        )
        .execute()
    )

    print("{0} cells updated.".format(result.get("updatedCells")))


def export_pandas_df_to_sheets(spreadsheet_id, file_path, sheet_name="Data"):
    df = pd.read_csv(file_path)
    body = {"values": df.values.tolist()}

    result = (
        spreadsheet_service.spreadsheets()
        .values()
        .append(
            spreadsheetId=spreadsheet_id,
            body=body,
            valueInputOption="USER_ENTERED",
            range=sheet_name,
        )
        .execute()
    )

    print("{0} cells appended.".format(result.get("updates").get("updatedCells")))


def clear_sheet(spreadsheet_id, sheets_names):
    body = {"ranges": sheets_names}

    result = (
            spreadsheet_service.spreadsheets()
            .values()
            .batchClear(
                spreadsheetId=spreadsheet_id,
                body=body
            ).execute()
    )

    print(f"{result.get('clearedRanges')}")


def read_range():
    range_name = "Sheet1!A1:H1"
    sheetId = "1JCEHwIa4ZzwAiKGmAnWGfbjeVCH_tWZF6MkIU0zICwM"
    result = (
        spreadsheet_service.spreadsheets()
        .values()
        .get(spreadsheetId=sheetId, range=range_name)
        .execute()
    )

    rows = result.get("values", [])
    print("{0} rows retrieved.".format(len(rows)))
    print("{0} rows retrieved.".format(rows))

    return rows


def write_range():
    spreadsheet_id = create_spreadsheet()
    range_name = "Sheet1!A1:H1"
    values = read_range()
    value_input_option = "USER_ENTERED"
    body = {"values": values}
    result = (
        spreadsheet_service.spreadsheets()
        .values()
        .update(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption=value_input_option,
            body=body,
        )
        .execute()
    )
    print("{0} cells updated.".format(result.get("updatedCells")))

    return spreadsheet_id


def read_ranges():
    spreadsheet_id = write_range()
    sheetId = "1JCEHwIa4ZzwAiKGmAnWGfbjeVCH_tWZF6MkIU0zICwM"
    range_names = ["Sheet1!A2:H21", "Sheet1!A42:H62"]
    result = (
        spreadsheet_service.spreadsheets()
        .values()
        .batchGet(spreadsheetId=sheetId, ranges=range_names)
        .execute()
    )
    ranges = result.get("valueRanges", [])
    print("{0} ranges retrieved.".format(len(ranges)))
    return ranges, spreadsheet_id


def write_ranges():
    values, spreadsheet_id = read_ranges()
    data = [
        {"range": "Sheet1!A2:H21", "values": values[0]["values"]},
        {"range": "Sheet1!A22:H42", "values": values[1]["values"]},
    ]
    body = {"valueInputOption": "USER_ENTERED", "data": data}
    print(body)
    result = (
        spreadsheet_service.spreadsheets()
        .values()
        .batchUpdate(spreadsheetId=spreadsheet_id, body=body)
        .execute()
    )
    print("{0} cells updated.".format(result.get("totalUpdatedCells")))