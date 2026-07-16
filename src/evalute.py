# Standard library
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Dict


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(args):
    model_choices = [
        "deepseek/deepseek-v4-flash",
        "qwen/qwen3.5-flash",
        "openai/gpt-oss-120b",
    ]
    model_to_metadata_files = {}
    for model in model_choices:
        safe_model_name = model.replace('/', '_')
        result_dir = Path(f'./results/llm/{safe_model_name}')
        model_to_metadata_files[model] = sorted([x for x in result_dir.glob('*.json')])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(args)
