from enum import Enum


class ErrorCategory(Enum):
    NOT_FOUND = 'not_found'
    INVALID_INPUT = 'invalid_input'
    EXTERNAL_API = 'external_api'
    FILE_ERROR = 'file_error'
    DATABASE = 'database'
    RATE_LIMIT = 'rate_limit'
    UNKNOWN = 'unknown'


def get_user_message(error: Exception, context: str = '') -> dict:
    """Inspect error type and message; return category, user_message, suggestion, and technical string."""
    err_str = str(error).lower()

    if 'not found' in err_str or '404' in err_str:
        return {
            'category': ErrorCategory.NOT_FOUND,
            'user_message': f'Could not find {context}. Please check the name and try again.' if context else 'Could not find it. Please check the name and try again.',
            'suggestion': 'Double-check the ticker symbol or coin ID is correct.',
            'technical': str(error),
        }
    if 'rate limit' in err_str or '429' in err_str:
        return {
            'category': ErrorCategory.RATE_LIMIT,
            'user_message': 'The data provider is temporarily limiting requests.',
            'suggestion': 'Wait 30 seconds and try again.',
            'technical': str(error),
        }
    if 'api_key' in err_str or 'authentication' in err_str:
        return {
            'category': ErrorCategory.INVALID_INPUT,
            'user_message': 'API authentication failed.',
            'suggestion': 'Check that your API key is correctly configured.',
            'technical': str(error),
        }
    if 'timeout' in err_str:
        return {
            'category': ErrorCategory.EXTERNAL_API,
            'user_message': 'The request took too long to complete.',
            'suggestion': 'Try again — this is usually temporary.',
            'technical': str(error),
        }
    if isinstance(error, ValueError):
        return {
            'category': ErrorCategory.INVALID_INPUT,
            'user_message': f'Invalid input for {context}.' if context else 'Invalid input.',
            'suggestion': 'Check the format of what you entered.',
            'technical': str(error),
        }
    if 'unsupported file' in err_str:
        return {
            'category': ErrorCategory.FILE_ERROR,
            'user_message': 'This file type is not supported.',
            'suggestion': 'Please upload a .txt or .pdf file.',
            'technical': str(error),
        }

    return {
        'category': ErrorCategory.UNKNOWN,
        'user_message': 'Something went wrong. Our team has been notified.',
        'suggestion': 'Try again in a moment.',
        'technical': str(error),
    }


def status_code_for_category(category: ErrorCategory) -> int:
    """Map ErrorCategory to HTTP status code for use in HTTPException."""
    if category == ErrorCategory.NOT_FOUND:
        return 404
    if category == ErrorCategory.RATE_LIMIT:
        return 429
    if category in (ErrorCategory.INVALID_INPUT, ErrorCategory.FILE_ERROR):
        return 400
    return 500
