from __future__ import print_function
from auth import spreadsheet_service, drive_service
import pandas as pd
import os
from dotenv import load_dotenv


def create():
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


def write_columns_names(file_path):
    spreadsheet_id = create()
    # values = ["id1", "id2", "id3", "students"]
    df = pd.read_csv(file_path, header=0)
    values = df.columns.to_list()

    range_name = f"Sheet1!A1:{chr(ord('A') + len(values) - 1)}1"
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
    #print("{0} cells updated.".format(result.get("updatedCells")))

    return spreadsheet_id

def write_columns(file_path, spreadsheet_id):
    # values = ["id1", "id2", "id3", "students"]
    df = pd.read_csv(file_path, header=0)
    values = df.columns.to_list()

    range_name = f"Sheet1!A1:{chr(ord('A') + len(values) - 1)}1"
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
    #print("{0} cells updated.".format(result.get("updatedCells")))


def write_range():
    spreadsheet_id = create()
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


def append():
    values, spreadsheet_id = read_ranges()
    data = [values[0]["values"], values[1]["values"]]
    body = {"valueInputOption": "USER_ENTERED", "data": data}
    result = (
        spreadsheet_service.spreadsheets()
        .values()
        .append(spreadsheetId=spreadsheet_id, body=body)
        .execute()
    )
    print("{0} cells updated.".format(result.get("totalUpdatedCells")))


def export_pandas_df_to_sheets(spreadsheet_id, file_path):
    df = pd.read_csv(file_path, header=0)
    # df = pd.DataFrame(
    # [
    # [0.21, 0.72, 0.67, 3],
    # [0.23, 0.78, 0.69, 3],
    # [0.32, -1, -1, 1],
    # [0.52, -1, 0.42, 2],
    # ],
    # columns=["id1", "id2", "id3", "students"],
    # )
    body = {"values": df.values.tolist()}

    result = (
        spreadsheet_service.spreadsheets()
        .values()
        .append(
            spreadsheetId=spreadsheet_id,
            body=body,
            valueInputOption="USER_ENTERED",
            range="Sheet1",
        )
        .execute()
    )
    #print("{0} cells appended.".format(result.get("updates").get("updatedCells")))
