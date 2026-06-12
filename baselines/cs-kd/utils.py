import os, logging
import sys
import time
import math
import shutil  # <-- 추가
import torch
import torch.nn as nn
import torch.nn.init as init


def set_logging_defaults(logdir, args):
    if os.path.isdir(logdir):
        res = input('"{}" exists. Overwrite [Y/n]? '.format(logdir))
        if res != 'Y':
            raise Exception('"{}" exists.'.format(logdir))
    else:
        os.makedirs(logdir)

    # set basic configuration for logging
    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, 'log.txt')),
                                  logging.StreamHandler(os.sys.stdout)])

    # log cmdline argumetns
    logger = logging.getLogger('main')
    logger.info(' '.join(os.sys.argv))
    logger.info(args)

# --- stty 사용 부분을 아래로 교체 ---
def _safe_term_width(default=120):
    """
    플랫폼/환경(Jupyter, 리다이렉트, Windows) 상관없이 안전하게 터미널 너비를 잡는다.
    """
    try:
        # Python 3.3+: 플랫폼 독립적
        return shutil.get_terminal_size(fallback=(default, 20)).columns
    except Exception:
        # 혹시 모를 예외 대비
        try:
            return int(os.environ.get("COLUMNS", default))
        except Exception:
            return default

term_width = _safe_term_width()
TOTAL_BAR_LENGTH = 86.0
last_time = time.time()
begin_time = last_time

# --- progress_bar 내부의 패딩/백스페이스 계산을 안전하게 ---
def progress_bar(current, total, msg=None):
    global last_time, begin_time, term_width

    # 터미널 크기가 바뀌었을 수 있어 매 스텝 갱신 (원하면 주석 처리)
    term_width = _safe_term_width()

    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    if rest_len < 0:
        rest_len = 0

    sys.stdout.write(' [')
    sys.stdout.write('=' * cur_len)
    sys.stdout.write('>')
    sys.stdout.write('.' * rest_len)
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)
    msg_str = ''.join(L)

    # 남은 칸 패딩을 음수 방지
    pad = max(0, term_width - int(TOTAL_BAR_LENGTH) - len(msg_str) - 3)
    if pad:
        sys.stdout.write(' ' * pad)

    # Go back to the center of the bar.
    back = max(0, term_width - int(TOTAL_BAR_LENGTH / 2))
    if back:
        sys.stdout.write('\b' * back)

    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()
def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
