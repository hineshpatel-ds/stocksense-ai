from src.security.upload_security import (
    get_file_extension,
    validate_file_size,
    validate_upload_filename,
    validate_upload_metadata,
)


def test_get_file_extension_lowercases_extension():
    assert get_file_extension("inventory.CSV") == ".csv"
    assert get_file_extension("report.XLSX") == ".xlsx"


def test_validate_upload_filename_accepts_csv():
    result = validate_upload_filename("inventory.csv")

    assert result.is_allowed is True
    assert result.extension == ".csv"


def test_validate_upload_filename_accepts_excel():
    result = validate_upload_filename("inventory.xlsx")

    assert result.is_allowed is True
    assert result.extension == ".xlsx"


def test_validate_upload_filename_rejects_missing_filename():
    result = validate_upload_filename("")

    assert result.is_allowed is False
    assert "Missing filename" in result.message


def test_validate_upload_filename_rejects_unsupported_extension():
    result = validate_upload_filename("malicious.exe")

    assert result.is_allowed is False
    assert "Unsupported file extension" in result.message


def test_validate_file_size_accepts_valid_file():
    result = validate_file_size(
        file_size_bytes=1024,
        max_size_mb=1,
    )

    assert result.is_allowed is True


def test_validate_file_size_rejects_empty_file():
    result = validate_file_size(
        file_size_bytes=0,
        max_size_mb=1,
    )

    assert result.is_allowed is False
    assert "empty" in result.message.lower()


def test_validate_file_size_rejects_large_file():
    result = validate_file_size(
        file_size_bytes=2 * 1024 * 1024,
        max_size_mb=1,
    )

    assert result.is_allowed is False
    assert "too large" in result.message.lower()


def test_validate_upload_metadata_accepts_valid_upload():
    result = validate_upload_metadata(
        filename="inventory.csv",
        file_size_bytes=1024,
    )

    assert result.is_allowed is True
    assert result.extension == ".csv"


def test_validate_upload_metadata_rejects_bad_extension():
    result = validate_upload_metadata(
        filename="inventory.txt",
        file_size_bytes=1024,
    )

    assert result.is_allowed is False