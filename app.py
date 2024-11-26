import streamlit as st
import pandas as pd
from io import BytesIO, StringIO
import re
import numpy as np

import os
import json
import io
from dotenv import load_dotenv
from googleapiclient.http import MediaIoBaseUpload

from googleapiclient.discovery import build
from google.oauth2 import service_account
load_dotenv()
# Constants
SCOPES = ['https://www.googleapis.com/auth/drive']
credentials_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
SERVICE_ACCOUNT_FILE = json.loads(credentials_json)# Path to your service account JSON file
REMOVED_FOLDER_ID = "1NWv0AjsOF-_5lmsEyL1q20liFWn1CtUk"  # Folder for "removed" files
SCRUBBED_FOLDER_ID = "1Ink3w5hpU5sAx9EvFmPu33W7HIbE1BIz"  # Fixed folder ID

# Authenticate Google Drive API
def authenticate():
    try:
        creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)
        st.info("Google Drive authenticated successfully.")
        return service
    except Exception as e:
        st.error(f"Google Drive authentication failed: {e}")
        raise e

service = authenticate()
def upload_to_drive_from_memory(file_name, file_data, folder_id):
    """
    Upload a file directly from memory to Google Drive.

    Args:
        file_name (str): Name of the file to upload.
        file_data (BytesIO): File data in memory.
        folder_id (str): ID of the Google Drive folder.
    """
    try:
        # Create a media file and metadata
        file_metadata = {
            'name': file_name,
            'parents': [folder_id],
        }

        # Wrap BytesIO object in MediaIoBaseUpload
        media = MediaIoBaseUpload(file_data, mimetype='text/csv', resumable=True)

        # Use Google Drive API to upload the file
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        st.success(f"Uploaded {file_name} to Google Drive successfully! File ID: {file.get('id')}")
        return file.get('id')
    except Exception as e:
        st.error(f"Failed to upload {file_name}: {e}")
        raise e


# Upload file to Google Drive

# Initialize session state to hold data and control reloads
if "list_file" not in st.session_state:
    st.session_state.list_file = None
if "log_files" not in st.session_state:
    st.session_state.log_files = []
if "log_filenames" not in st.session_state:
    st.session_state.log_filenames = []
if "conditions" not in st.session_state:
    st.session_state.conditions = []
    

def clean_number(phone):
    """
    Clean and standardize phone numbers consistently.
    
    Args:
        phone: Input phone number (string or numeric)
    
    Returns:
        Cleaned 10-digit phone number string
    """
    # Convert to string and remove all non-digit characters
    phone = str(phone)  # Convert to string
    phone = re.sub(r'\D', '', phone)  # Remove non-digit characters
    if phone.startswith('1') and len(phone) > 10:  # Remove leading '1' if applicable
        phone = phone[1:]
    return phone
def clean_number_to_text(df):
    """
    Converts all numeric columns to string with integer formatting (no decimals).
    Simulates Excel's =TEXT(A1, "0") functionality.
    
    Args:
        df (pd.DataFrame): DataFrame to process.

    Returns:
        pd.DataFrame: Processed DataFrame with numeric columns converted to strings.
    """
    for col in df.columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Convert numeric column to formatted strings without decimals
            df[col] = df[col].apply(lambda x: f"{int(x)}" if pd.notnull(x) else "")
    return df

def process_files(log_dfs, list_df, conditions, log_filenames):
    """
    Process log files by removing specific phone numbers based on conditions.
    
    Args:
        log_dfs (list): List of DataFrames for log files.
        list_df (DataFrame): DataFrame for the list file.
        conditions (list): List of filtering conditions.
        log_filenames (list): Names of the log files.
    
    Returns:
        tuple: Updated list DataFrame, updated log DataFrames, removed list DataFrame.
    """
    # Normalize list file phone numbers
    list_df["Phone"] = list_df["Phone"].astype(str).apply(clean_number)

    # Compute occurrences in the list file
    list_occurrences = (
        list_df.groupby(["Log Type", "Phone"])
        .size()
        .reset_index(name="occurrence")
    )
    list_df = pd.merge(list_df, list_occurrences, on=["Log Type", "Phone"], how="left")

    # Initialize removed data containers
    removed_from_list = pd.DataFrame()
    updated_log_dfs = []

    # Parse conditions into a dictionary
    parsed_conditions = {cond["type"].title(): cond["threshold"] for cond in conditions}

    # Identify numbers to remove based on conditions
    for cond_type, threshold in parsed_conditions.items():
        matching_numbers = list_df.loc[
            (list_df["Log Type"].str.title() == cond_type) &
            (list_df["occurrence"] >= threshold), 
            "Phone"
        ].unique()

        # If no matching numbers, skip
        if len(matching_numbers) == 0:
            continue

        # Add matching numbers to the removed list
        current_removed = list_df[list_df["Phone"].isin(matching_numbers)]
        removed_from_list = pd.concat([removed_from_list, current_removed])

        # Remove these numbers from the list DataFrame
        list_df = list_df[~list_df["Phone"].isin(matching_numbers)]

    # Deduplicate removed list
    removed_from_list = removed_from_list.drop_duplicates()

    # Ensure unique cleaned phone numbers for removal
    cleaned_phones_to_remove = [clean_number(phone) for phone in removed_from_list["Phone"].unique()]
    print("Numbers to remove (cleaned):", cleaned_phones_to_remove)

    # Process each log file
    for log_df, filename in zip(log_dfs, log_filenames):
        processed_log_df = log_df.copy()

        # Normalize column names
        processed_log_df.columns = processed_log_df.columns.str.strip().str.lower()

        # Ensure all columns are strings to avoid data type mismatches
        processed_log_df = processed_log_df.astype(str)
       #processed_log_df=clean_number_to_text(processed_log_df)
        # Identify potential phone number columns
        phone_columns = [
            col for col in processed_log_df.columns
            if any(phrase in col.lower() for phrase in ['mobile', 'phone', 'number', 'tel', 'contact','ph'])
        ]
        print(f"\nProcessing file: {filename}")
        print("Phone columns detected:", phone_columns)

        # Debugging: Show the first few rows of phone columns
        print(f"Sample data from phone columns in {filename}:")
        print(processed_log_df[phone_columns].head())

        # If no phone columns found, keep the original DataFrame
        if not phone_columns:
            updated_log_dfs.append(processed_log_df)
            continue
        
        # Process each phone column
        for col in phone_columns:
            # Clean the column's phone numbers
            processed_log_df[col] = processed_log_df[col].astype(str).apply(
        lambda x: f"{int(float(x))}" if x.replace(".", "").isdigit() else x
    )
            cleaned_column = processed_log_df[col].apply(clean_number)
        
            
            # Debugging: Print cleaned values for this column
            print(f"Cleaned values for column {col}:")
            print(cleaned_column.unique())

            # Identify rows to remove
            remove_mask = cleaned_column.isin(cleaned_phones_to_remove)
            print(f"Column: {col}, Rows to be removed: {remove_mask.sum()}")

            # Debugging: Show matching rows before removal
            if remove_mask.any():
                print(f"Rows to be removed from column {col}:")
                print(processed_log_df.loc[remove_mask, col])

            # Replace matching numbers with an empty string
            processed_log_df.loc[remove_mask, col] = ''

        # Add the processed log DataFrame to the list
        updated_log_dfs.append(processed_log_df)

    return list_df, updated_log_dfs, removed_from_list

st.title("Log and List File Processor")

# File upload section
st.header("Upload Files")

# List file uploader
uploaded_list_file = st.file_uploader("Upload List File (CSV)", type="csv")
if uploaded_list_file:
    st.session_state.list_file = pd.read_csv(uploaded_list_file)
    st.session_state.list_file_name = os.path.splitext(uploaded_list_file.name)[0]
# Log files uploader
uploaded_log_files = st.file_uploader("Upload Log Files (CSV)", type="csv", accept_multiple_files=True)
if uploaded_log_files:
    st.session_state.log_files = [pd.read_csv(file) for file in uploaded_log_files]
    st.session_state.log_filenames = [file.name for file in uploaded_log_files]

# Ensure files are loaded in session state
if st.session_state.list_file is not None and st.session_state.log_files:
    list_file = st.session_state.list_file
    log_files = st.session_state.log_files
    log_filenames = st.session_state.log_filenames

    # Display list file preview
    st.write("List File Preview:")
    st.dataframe(list_file)

    # Display log files preview
    st.write(f"{len(log_files)} Log Files Uploaded")

    # Add user conditions
    st.header("Set Conditions")
    condition_type = st.text_input("Enter Condition Type (e.g., voicemail, call):").capitalize()
    condition_threshold = st.number_input("Enter Threshold (Occurrence Count):", min_value=1, step=1)

    if st.button("Add Condition"):
        # Initialize conditions in session state if not exists
        if "conditions" not in st.session_state:
            st.session_state.conditions = []
        
        st.session_state.conditions.append({"type": condition_type, "threshold": condition_threshold})
        st.success(f"Condition added: {condition_type} with threshold {condition_threshold}")

    st.write("Current Conditions:")
    if st.session_state.conditions:
        for idx, cond in enumerate(st.session_state.conditions):
            col1, col2 = st.columns([3, 1])  # Create columns for layout
            with col1:
                st.write(f"{cond['type']} - min Count: {cond['threshold']}")
            with col2:
                if st.button("Remove", key=f"remove_{idx}"):
                    # Remove the condition from the list
                    st.session_state.conditions.pop(idx)
                    st.success(f"Removed condition: {cond['type']} with threshold {cond['threshold']}")
                    
                    # Force a rerun to update the UI immediately
                    st.experimental_rerun()
    else:
        st.info("No conditions added yet.")

    # Process files when the user clicks the button
    import io

# Process files when the user clicks the button
if st.button("Process Files"):
    try:
        if not st.session_state.get("conditions"):
            st.error("Please add at least one condition before processing.")
        else:
            updated_list_df, updated_log_dfs, removed_list_df = process_files(
                log_files, list_file, st.session_state.conditions, log_filenames
            )
            updated_list_df = updated_list_df.drop_duplicates()

            st.session_state.updated_list_df = updated_list_df
            st.session_state.updated_log_dfs = updated_log_dfs
            st.session_state.removed_list_df = removed_list_df
            list_file_name = st.session_state.list_file_name
            # Upload files directly from memory
            # Updated list file
            updated_list_io = BytesIO()
            updated_list_df.to_csv(updated_list_io, index=False)
            updated_list_io.seek(0)
            upload_to_drive_from_memory(f"Updated_{list_file_name}.csv", updated_list_io, REMOVED_FOLDER_ID)

            removed_list_io = BytesIO()
            removed_list_df.to_csv(removed_list_io, index=False)
            removed_list_io.seek(0)
            upload_to_drive_from_memory(f"Scrubbed_{list_file_name}_Removed.csv", removed_list_io, REMOVED_FOLDER_ID)


            # Log files
            for i, log_file_name in enumerate(log_filenames):
                scrubbed_log_io = BytesIO()
                st.session_state.updated_log_dfs[i].to_csv(scrubbed_log_io, index=False)
                scrubbed_log_io.seek(0)
                upload_to_drive_from_memory(f"Scrubbed_{log_file_name}", scrubbed_log_io, SCRUBBED_FOLDER_ID)

            st.success("Files processed and uploaded successfully!")
    except Exception as e:
        st.error(f"Error processing files: {e}")


# Display results if processing is complete
if "updated_list_df" in st.session_state:
    st.subheader("Updated List File")
    st.dataframe(st.session_state.updated_list_df)

    st.subheader("Numbers Removed from List File")
    st.dataframe(st.session_state.removed_list_df)

    # Dynamic download buttons
    # Prepare list file download
    updated_list_csv = st.session_state.updated_list_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Updated List File",
        data=updated_list_csv,
        file_name="Updated_List_File.csv",
        mime="text/csv",
    )

    # Prepare removed list download
    removed_list_csv = st.session_state.removed_list_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Scrubbed List File Removed",
        data=removed_list_csv,
        file_name="Scrubbed_Listname_Removed.csv",
        mime="text/csv",
    )

    # Dynamic download buttons for log files
    st.subheader("Log Files")
    for i, log_file_name in enumerate(st.session_state.log_filenames):
        # Scrubbed log file
        scrubbed_log_csv = st.session_state.updated_log_dfs[i].to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download Scrubbed Log File ({log_file_name})",
            data=scrubbed_log_csv,
            file_name=f"Scrubbed_{log_file_name}",
            mime="text/csv",
        )

        # Removed log rows
        # Find and extract rows that were removed
        original_log_df = st.session_state.log_files[i]
        current_log_df = st.session_state.updated_log_dfs[i]
        
    
