import logging
import sys
from typing import Literal, Optional

RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
GREY = '\033[37m'
RESET = '\033[0m'

format_string = '%(asctime)s %(levelname)s %(message)s'

FORMATS = {
    logging.DEBUG: GREY + format_string + RESET,
    logging.INFO: GREEN + format_string + RESET,
    logging.WARNING: YELLOW + format_string + RESET,
    logging.ERROR: RED + format_string + RESET
}


def _get_format(level) -> str:
    return FORMATS[level]


class CustomerFormatter(logging.Formatter):

    def format(self, record):
        format = _get_format(record.levelno)
        formatter = logging.Formatter(format)
        return formatter.format(record)


_logger = logging.getLogger('Analyzer')
_logger.setLevel(logging.DEBUG)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(CustomerFormatter())
_logger.addHandler(_handler)


def set_debug_level():
    _logger.setLevel(logging.DEBUG)


def set_info_level():
    _logger.setLevel(logging.INFO)


def set_warning_level():
    _logger.setLevel(logging.WARNING)


def _print_msg(msg: str, level: Literal['debug', 'info', 'warning', 'error'], template: Optional[str] = None):
    if template is not None:
        msg = template.format(msg=msg)

    msg = f'\n{msg}\n'
    if level == 'debug':
        _logger.debug(msg)
    elif level == 'info':
        _logger.info(msg)
    elif level == 'warning':
        _logger.warning(msg)
    elif level == 'error':
        _logger.error(msg)
    else:
        raise ValueError(f'logger/_print_msg: unexpected level = {level}')


def log_debug(msg: str, template: Optional[str] = None):
    _print_msg(msg, 'debug', template)


def log_info(msg: str, template: Optional[str] = None):
    _print_msg(msg, 'info', template)


def log_warning(msg: str, template: Optional[str] = None):
    _print_msg(msg, 'warning', template)


def log_error(msg: str, template: Optional[str] = None):
    _print_msg(msg, 'error', template)


class WithLogger:
    _log_msg_template: str

    @classmethod
    def _get_log_template(cls) -> str:
        return cls._log_msg_template

    @classmethod
    def log_debug(cls, msg: str):
        return log_debug(msg, cls._get_log_template())

    @classmethod
    def log_info(cls, msg: str):
        return log_info(msg, cls._get_log_template())

    @classmethod
    def log_warning(cls, msg: str):
        return log_warning(msg, cls._get_log_template())

    @classmethod
    def log_error(cls, msg: str):
        return log_error(msg, cls._get_log_template())