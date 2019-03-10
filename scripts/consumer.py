import os
import subprocess
import time
import argparse
import sys  # Add `source` module (needed when it runs via terminal)
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from source.utils import chdir, create_logger


def next_in(file_name):
    """ Get the first file line - truncate to the current pointer position."""
    with open(file_name, "r+") as f:
        line = f.readline()
        remaining_lines = f.readlines()
        f.seek(0)
        f.writelines(remaining_lines)
        f.truncate()
    return line


def execute(command):
    """ Execute a python script and waits for it to finish (security issue). """
    handler, = logger.handlers
    subprocess.run(f'{args.python} {command}'.split(), stderr=handler.stream, cwd=os.getcwd(), check=True)


def run_consumer(queue):
    """ Run program in the infinite loop (a daemon process). """
    while True:
        try:
            command = next_in(queue)

            if command:
                logger.info(f'Program execute the command: {command}')
                execute(command)
                logger.info(f'Correct run: {command}')
            else:
                time.sleep(10)

        except subprocess.CalledProcessError:
            logger.error(f'Program can not execute the command')


if __name__ == "__main__":
    chdir(to='ROOT')
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', required=True, help='The queue with tasks to execute')
    parser.add_argument('--python', required=True, help='The interpreter path')
    parser.add_argument('--log_file', required=True, help='The log file')
    parser.add_argument('--log_level', type=int, default=20, help='The log level (default set to INFO)')
    args = parser.parse_args()

    logger = create_logger(args.log_file, args.log_level, name='consumer')

    logger.info('The consumer has been started')
    logger.info(f'Execute the queue: {args.queue}')

    run_consumer(queue=args.queue)
