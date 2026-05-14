from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".xlsx", ".xls"}
DEFAULT_MAX_UPLOAD_SIZE_MB = 10


@dataclass
class UploadSecurityResult:
    """
    Result of upload security validation.
    """

    is_allowed: bool
    message: str
    extension: str | None = None
    file_size_mb: float | None = None


def get_max_upload_size_mb() -> int:
    """
    Get maximum allowed upload size from environment variable.

    Default:
        10 MB
    """

    raw_value = os.getenv(
        "STOCKSENSE_MAX_UPLOAD_SIZE_MB",
        str(DEFAULT_MAX_UPLOAD_SIZE_MB),
    )

    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_MAX_UPLOAD_SIZE_MB

    return max(value, 1)


def get_file_extension(filename: str | None) -> str:
    """
    Get lowercase file extension from filename.
    """

    if not filename:
        return ""

    return Path(filename).suffix.lower()


def validate_upload_filename(filename: str | None) -> UploadSecurityResult:
    """
    Validate uploaded filename and extension.
    """

    if not filename or not filename.strip():
        return UploadSecurityResult(
            is_allowed=False,
            message="Missing filename. Please upload a valid CSV or Excel file.",
        )

    extension = get_file_extension(filename)

    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        return UploadSecurityResult(
            is_allowed=False,
            message=(
                f"Unsupported file extension '{extension}'. "
                "Allowed file types are CSV, XLSX, and XLS."
            ),
            extension=extension,
        )

    return UploadSecurityResult(
        is_allowed=True,
        message="Filename and extension are allowed.",
        extension=extension,
    )


def validate_file_size(
    file_size_bytes: int,
    max_size_mb: int | None = None,
) -> UploadSecurityResult:
    """
    Validate uploaded file size.
    """

    max_allowed_mb = max_size_mb or get_max_upload_size_mb()
    file_size_mb = file_size_bytes / (1024 * 1024)

    if file_size_bytes <= 0:
        return UploadSecurityResult(
            is_allowed=False,
            message="Uploaded file is empty.",
            file_size_mb=round(file_size_mb, 4),
        )

    if file_size_mb > max_allowed_mb:
        return UploadSecurityResult(
            is_allowed=False,
            message=(
                f"Uploaded file is too large: {file_size_mb:.2f} MB. "
                f"Maximum allowed size is {max_allowed_mb} MB."
            ),
            file_size_mb=round(file_size_mb, 4),
        )

    return UploadSecurityResult(
        is_allowed=True,
        message="File size is allowed.",
        file_size_mb=round(file_size_mb, 4),
    )


def validate_upload_metadata(
    filename: str | None,
    file_size_bytes: int,
) -> UploadSecurityResult:
    """
    Validate uploaded file metadata before processing.

    Checks:
    - filename exists
    - extension is allowed
    - file size is acceptable
    """

    filename_result = validate_upload_filename(filename)

    if not filename_result.is_allowed:
        return filename_result

    size_result = validate_file_size(file_size_bytes)

    if not size_result.is_allowed:
        return size_result

    return UploadSecurityResult(
        is_allowed=True,
        message="Upload metadata passed security checks.",
        extension=filename_result.extension,
        file_size_mb=size_result.file_size_mb,
    )