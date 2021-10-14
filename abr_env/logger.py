import logging


def set_logging(s_level='info', log_to='print'):
    if s_level == 'debug':
        level = logging.DEBUG
    elif s_level == 'info':
        level = logging.INFO
    elif s_level == 'warning':
        level = logging.WARNING
    elif s_level == 'error':
        level = logging.ERROR
    else:
        raise ValueError('Unknown logging level ' + s_level)
    if log_to == 'print':
        logging.basicConfig(level=level)
    else:
        logging.basicConfig(filename=log_to, level=level)


def debug(msg):
    logging.debug(msg)


def info(msg):
    logging.info(msg)


def warn(msg):
    logging.warning(msg)


def error(msg):
    logging.error(msg)


def exception(msg, *args, **kwargs):
    logging.exception(msg, *args, **kwargs)
