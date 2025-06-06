import os
import subprocess
import sys
import requests
from typing import Optional, Tuple
from dotenv import load_dotenv
from pathlib import Path

from .Model import Model

load_dotenv()


class DeepSeekV3Model(Model):
    def __init__(self):
        super().__init__()
        raw_url = os.getenv("LLAMA_SERVER_URL", "").strip()
        self.server_url = raw_url if raw_url and ":" in raw_url else None
        self.llama_bin_path = '/data1/llama.cpp/bin'
        self.llama_server_path = '/data1/GGUF'
        self.model_path = '/data1/GGUF/DeepSeek-V3-0324-UD-Q2_K_XL/DeepSeek-V3-0324-UD-Q2_K_XL.gguf'
        self.gpu_layers = '40'

    def generate_response(
        self,
        prompt: str,
        submission_file: Path,
        solution_file: Optional[Path] = None,
        scope: Optional[str] = None,
        question_num: Optional[int] = None,
        test_output: Optional[Path] = None,
        llama_mode: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        """
        Generate a model response using the prompt and assignment files.

        Args:
            prompt (str): The input prompt provided by the user.
            submission_file (Path): Path Object pointing to the submission file.
            solution_file (Path): Path Object pointing to the solution file.
            scope (Optional[str]): Optional scope to use for this model.
            test_output (Optional[Path]): Path Object pointing to the test output file.
            llama_mode (Optional[str]): Optional mode to invoke llama.cpp in.
            question_num (Optional[int]): An optional question number to target specific content.

        Returns:
            Optional[Tuple[str, str]]: A tuple containing the prompt and the model's response,
                                       or None if the response was invalid.
        """
        if llama_mode == 'server':
            if not self.server_url:
                raise RuntimeError("Error: Environment variable LLAMA_SERVER_URL not set")
            response = self._get_response_server(prompt)
        else:
            prompt = f"'{prompt}'"
            response = self._get_response_cli(prompt)

        # Remove prompt from response
        if response.startswith(prompt):
            tail = response[len(prompt):]
            if tail.startswith("\n"):
                tail = tail[1:]
            response = tail

        response = response.strip()

        # Remove end of response marker
        end_marker = "[end of text]"
        if response.endswith(end_marker):
            response = response[: -len(end_marker)]
            response = response.strip()

        return prompt, response

    def _get_response_server(
            self,
            prompt: str,
    ) -> str:
        """
        Generate a model response using the prompt

        Args:
            prompt (str): The input prompt provided by the user.

        Returns:
            str: A tuple containing the model response or None if the response was invalid.
        """
        url = f"http://{self.server_url}/v1/completions"

        payload = {
            "prompt": prompt,
        }

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError("ERROR: Request to llama-server failed:", str(e), file=sys.stderr, flush=True)

        data = response.json()

        try:
            model_output = data["choices"][0]["text"]
        except (KeyError, IndexError):
            print("ERROR: Unexpected JSON format from llama-server:", data, file=sys.stderr, flush=True)
            model_output = ''

        return model_output

    def _get_response_cli(
            self,
            prompt: str,
    ) -> str:
        """
        Generate a model response using the prompt

        Args:
            prompt (str): The input prompt provided by the user.

        Returns:
            str: The model response or None if the response was invalid.
        """
        # Need to add quotes to the prompt since prompts are multiline

        cmd = [
            "./llama-cli",
            "-m", self.model_path,
            "--n-gpu-layers", self.gpu_layers,
            "-no-cnv",
            "-p", prompt
        ]

        try:
            completed = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.llama_bin_path,
                timeout=300
            )
        except subprocess.TimeoutExpired as e:
            # If the process hangs for more than 5 minutes, print whatever has been captured so far
            print("ERROR: llama-cli timed out after 5 minutes.", file=sys.stdout, flush=True)
            print("Partial stdout:", e.stdout, file=sys.stdout, flush=True)
            print("Partial stderr:", e.stderr, file=sys.stdout, flush=True)
            raise
        except subprocess.CalledProcessError as e:
            # If llama-cli returns a non-zero exit code, print its stdout/stderr and re-raise
            print("ERROR: llama-cli returned non-zero exit code.", file=sys.stdout, flush=True)
            print("llama-cli stdout:", e.stdout, file=sys.stdout, flush=True)
            print("llama-cli stderr:", e.stderr, file=sys.stdout, flush=True)
            raise RuntimeError(
                f"llama.cpp failed (code {e.returncode}): {e.stderr.strip()}"
            )

        # Decode with 'replace' so invalid UTF-8 bytes become U+FFFD
        return completed.stdout.decode('utf-8', errors='replace')

