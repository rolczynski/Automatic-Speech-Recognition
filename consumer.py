import os
import time
import argparse
import traceback
from source.deepspeech import DeepSpeech
from source.configuration import Configuration
from contextlib import redirect_stdout


def __get_first_line(queue):
    """ Get the first file line - truncate to the current pointer position."""
    with open(queue, "r+") as f:
        line = f.readline()
        remaining_lines = f.readlines()
        f.seek(0)
        f.writelines(remaining_lines)
        f.truncate()
    return line


def __update_parameters(configuration, parameters):
    """ Eval custom parameters which overwrite the template setup. """
    for parameter in parameters:
        exec('configuration.' + parameter)


def __create_experiment_dir(configuration):
    """ Create experiment dir """
    os.makedirs(configuration.exp_dir)


def __run_program(configuration_line):
    """ Run DeepSpeech - save log file. """
    configuration_file_path, *parameters = configuration_line.split('|')
    configuration = Configuration(configuration_file_path)
    __update_parameters(configuration, parameters)
    __create_experiment_dir(configuration)

    deepspeech_output = os.path.join(configuration.exp_dir, 'program.out')
    with open(deepspeech_output, 'w') as f:
        with redirect_stdout(f):
            ds = DeepSpeech(configuration)
            ds.train()
            ds.save()


def __run_consumer(queue, program_output):
    """ Run program in the infinite loop and save logs """
    while True:
        try:
            configuration_line = __get_first_line(queue)
            if configuration_line:
                __run_program(configuration_line)
                print(f'Correct run: {configuration_line}')
            else:
                time.sleep(10)

        except Exception as exception:
            print(f'Problem occurs: {configuration_line}')
            print(traceback.format_exc())

        finally:
            program_output.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', help='Task queue to execute')
    args = parser.parse_args()

    name, extension = os.path.basename(args.queue).split('.')
    program_output = os.path.join('experiments', f'consumer-{name}.out')
    with open(program_output, 'w') as f:
        with redirect_stdout(f):
            __run_consumer(queue=args.queue, program_output=f)
