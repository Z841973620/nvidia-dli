import json
import os 
from subprocess import check_output

def runtest():
    score = 0
    message = ''
    error_message = '''There was an error assessing your code.
                       Please make sure you are
                       following the assessment directions
                       and try again'''

    this_path = os.path.dirname(os.path.realpath(__file__))

    try:
        output = check_output([os.path.join(this_path, 'assess_student_code.sh')])
        is_correct = output.decode("utf-8").strip()
    except:
        return error_message

    if is_correct == 'True':
        score += 100
        message += 'Your code produced the correct output.\n'
    else:
        message += 'Your code did not produce the correct output.\n'

    if score == 100:
        message += 'Congratulations, you passed!'
    else:
        message += 'You did not pass, please try again.'

    print(message)
