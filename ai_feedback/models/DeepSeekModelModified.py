import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import ollama
import requests
from dotenv import load_dotenv

from .Model import Model
from ..helpers.model_options_helpers import cast_to_type, ollama_option_schema

load_dotenv()

class DeepSeekModelModified(Model):
    """
    Backends:
        - ollama: Use Ollama API (for models like deepseek-r1:70b)
        - llama: Use llama.cpp (CLI or server mode, for models like deepseek-v3)

    Examples:
        # DeepSeek R1 via Ollama
        model = DeepSeekModelModified(model_name="deepseek-r1:70b", backend="ollama")

        # DeepSeek V3 via llama.cpp server
        model = DeepSeekModelModified(model_name="deepseek-v3", backend="llama", llama_mode="server")

        # DeepSeek V3 via llama.cpp CLI
        model = DeepSeekModelModified(model_name="deepseek-v3", backend="llama", llama_mode="cli")
    """

    def __init__(
        self,
        model_name: str = "deepseek-r1:70b",
        backend: str = "ollama",
        llama_mode: str = "cli",
    ) -> None:
        """
        Initializes the unified DeepSeekModel with configurable backend.

        Args:
            model_name (str): The model name/identifier to use.
            backend (str): Backend to use - "ollama" or "llama".
            llama_mode (str): Mode for llama.cpp backend - "cli" or "server".
        """
        super().__init__(model_name)
        self.backend = backend
        self.model_name = model_name
        self.llama_mode = llama_mode

        # Load llama.cpp configuration
        if backend == "llama":
            self.gpu_layers = os.getenv('GPU_LAYERS', '40')
            self.llama_cli_path = os.getenv('LLAMA_CLI_PATH', '')
            self.llama_model_path = os.getenv('LLAMA_MODEL_PATH', '')
            self.llama_server_url = os.getenv('LLAMA_SERVER_URL', '').strip()

    def generate_response(
        self,
        prompt: str,
        submission_file: Path,
        system_instructions: str,
        model_options: Optional[dict] = None,
        question: Optional[str] = None,
        solution_file: Optional[Path] = None,
        test_output: Optional[Path] = None,
        scope: Optional[str] = None,
        llama_mode: Optional[str] = None,
        json_schema: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        """
        Generate a model response using the prompt and assignment files.

        Args:
            prompt (str): The input prompt provided by the user.
            submission_file (Optional[Path]): The path to the submission file.
            solution_file (Optional[Path]): The path to the solution file.
            test_output (Optional[Path]): The path to the test output file.
            scope (Optional[str]): The scope to use for generating the response.
            question (Optional[str]): An optional question to target specific content.
            system_instructions (str): Instructions for the model.
            llama_mode (Optional[str]): Optional mode to invoke llama.cpp in (overrides constructor).
            json_schema (Optional[str]): Optional json schema to use.
            model_options (Optional[dict]): The optional model options to use for generating the response.

        Returns:
            Optional[Tuple[str, str]]: A tuple containing the prompt and the model's response,
                                       or None if the response was invalid.
        """
        # Load JSON schema if provided
        schema = self._load_json_schema(json_schema)

        # Use llama_mode from parameter if provided, otherwise use constructor value
        effective_llama_mode = llama_mode if llama_mode else self.llama_mode

        # Route to appropriate backend
        if self.backend == "ollama":
            response = self._generate_with_ollama(prompt, system_instructions, model_options, schema)
        elif self.backend == "llama":
            response = self._generate_with_llama(prompt, system_instructions, model_options, schema, effective_llama_mode)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}. Use 'ollama' or 'llama'.")

        return response

    def _load_json_schema(self, json_schema: Optional[str]) -> Optional[dict]:
        """
        Load JSON schema from file if provided.

        Args:
            json_schema (Optional[str]): Path to JSON schema file.

        Returns:
            Optional[dict]: Loaded schema or None.
        """
        if json_schema:
            schema_path = Path(json_schema)
            if not schema_path.exists():
                raise FileNotFoundError(f"JSON schema file not found: {schema_path}")
            with open(schema_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _generate_with_ollama(
        self,
        prompt: str,
        system_instructions: str,
        model_options: Optional[dict],
        schema: Optional[dict],
    ) -> Optional[Tuple[str, str]]:
        """
        Generate response using Ollama backend.

        Args:
            prompt (str): User prompt.
            system_instructions (str): System instructions.
            model_options (Optional[dict]): Model options.
            schema (Optional[dict]): JSON schema for structured output.

        Returns:
            Optional[Tuple[str, str]]: Tuple of (prompt, response) or None.
        """
        model_options = cast_to_type(ollama_option_schema, model_options)

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt},
            ],
            format=schema['schema'] if schema else None,
            options=model_options if model_options else None,
        )

        if not response or "message" not in response or "content" not in response["message"]:
            print("Error: Invalid or empty response from Ollama.")
            return None

        return prompt, response["message"]["content"]

    def _generate_with_llama(
        self,
        prompt: str,
        system_instructions: str,
        model_options: Optional[dict],
        schema: Optional[dict],
        llama_mode: str,
    ) -> Optional[Tuple[str, str]]:
        """
        Generate response using llama.cpp backend (CLI or server).

        Args:
            prompt (str): User prompt.
            system_instructions (str): System instructions.
            model_options (Optional[dict]): Model options.
            schema (Optional[dict]): JSON schema for structured output.
            llama_mode (str): "cli" or "server".

        Returns:
            Optional[Tuple[str, str]]: Tuple of (full_prompt, response) or None.
        """
        # Combine system instructions and prompt
        full_prompt = f"{system_instructions}\n{prompt}"

        if llama_mode == 'server':
            self._ensure_env_vars('llama_server_url')
            response = self._get_response_server(full_prompt, model_options, schema)
        else:  # cli mode
            self._ensure_env_vars('llama_model_path', 'llama_cli_path')
            response = self._get_response_cli(full_prompt, model_options, schema)

        response = response.strip()

        # Remove end of response marker
        end_marker = "[end of text]"
        if response.endswith(end_marker):
            response = response[: -len(end_marker)]
            response = response.strip()

        return full_prompt, response

    def _ensure_env_vars(self, *names):
        """
        Ensure that each of the given attribute names exists on self and is truthy.

        Args:
            *names (str): One or more attribute names to validate.

        Raises:
            RuntimeError: If any of the specified attributes is missing or has a falsy value.
        """
        missing = [n for n in names if not getattr(self, n, None)]
        if missing:
            raise RuntimeError(
                f"Error: Required configuration variable(s) {', '.join(missing)} not set. "
                f"Please check your environment variables."
            )

    def _get_response_server(
        self, prompt: str, model_options: Optional[dict] = None, schema: Optional[dict] = None
    ) -> str:
        """
        Generate a model response using llama.cpp server.

        Args:
            prompt (str): The input prompt provided by the user.
            schema (Optional[dict]): Optional schema provided by the user.
            model_options (Optional[dict]): The optional model options to use for generating the response.

        Returns:
            str: The model response.
        """
        url = f"{self.llama_server_url}/v1/completions"

        payload = {"prompt": prompt, **(model_options or {})}

        if schema:
            raw_schema = schema.get("schema", schema)
            payload["json_schema"] = raw_schema

        try:
            response = requests.post(url, json=payload, timeout=3000)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"ERROR: Request to llama-server failed: {str(e)}")

        data = response.json()

        try:
            model_output = data["choices"][0]["text"]
        except (KeyError, IndexError):
            print("ERROR: Unexpected JSON format from llama-server:", data, file=sys.stderr, flush=True)
            model_output = ''

        return model_output

    def _get_response_cli(
        self, prompt: str, model_options: Optional[dict] = None, schema: Optional[dict] = None
    ) -> str:
        """
        Generate a model response using llama.cpp CLI.

        Args:
            prompt (str): The input prompt provided by the user.
            schema (Optional[dict]): Optional schema provided by the user.
            model_options (Optional[dict]): The optional model options to use for generating the response.

        Returns:
            str: The model response.
        """
        cmd = [
            self.llama_cli_path,
            "-m",
            self.llama_model_path,
            "--n-gpu-layers",
            self.gpu_layers,
            "--single-turn",
            "--no-display-prompt",
        ]

        if schema:
            raw_schema = schema["schema"] if "schema" in schema else schema
            cmd += ["--json-schema", json.dumps(raw_schema)]

        if model_options:
            for key, value in model_options.items():
                cmd += ["--" + key, str(value)]

        try:
            completed = subprocess.run(
                cmd, input=prompt.encode(), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300
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
            raise RuntimeError(f"llama.cpp failed (code {e.returncode}): {e.stderr.strip()}")

        # Decode with 'replace' so invalid UTF-8 bytes become U+FFFD
        return completed.stdout.decode('utf-8', errors='replace')
