"""Persistent OpenNARS for Applications agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
import re
import subprocess
from threading import Thread
from time import monotonic
from typing import Literal, Sequence, TextIO

from adaptive_sorting.agents.base_agent import BaseAgent
from adaptive_sorting.env.sorting_task_env import (
    Action,
    Observation,
    context_light_states,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ONA_BINARY = (
    PROJECT_ROOT / "third_party" / "OpenNARS-for-Applications" / "NAR-c256-t240"
)
RESPONSE_END = "done with 0 additional inference steps."
DEFAULT_ONA_TIMEOUT = 5.0
DEFAULT_ONA_STARTUP_TIMEOUT = 20.0
GOAL = "correct_sorting"
POSITIVE_FEEDBACK_TRUTH = "{1.0 0.9}"
NEGATIVE_FEEDBACK_TRUTH = "{0.0 0.9}"
# ONA uses a strict random < chance comparison, so 1.0 has a 1/32768 miss case.
MOTOR_BABBLING_CHANCE = 1.0001
# Mirrors MOTOR_BABBLING_SUPPRESSION_THRESHOLD in ONA's Config.h; only labels decision sources.
MOTOR_BABBLING_SUPPRESSION_THRESHOLD = 0.55
FLAT_ENCODING = "flat"
RELATIONAL_SORTING_ENCODING = "relational_sorting"
InteractionEncoding = Literal["flat", "relational_sorting"]
PARAMETERIZED_OPERATION = "place"
DECISION_PATTERN = re.compile(
    r"^decision expectation=(?P<expectation>[0-9.]+) implication: "
    r"(?P<implication>.*)$"
)
RULE_FAMILIES = (
    "color",
    "L1",
    "L2",
    "color+L1",
    "color+L2",
    "L1+L2",
    "color+L1+L2",
)
DecisionSource = Literal["learned_rule", "motor_babbling", "unknown"]


@dataclass(frozen=True)
class DecisionDiagnostic:
    """Provenance of the operation selected for one ONA goal."""

    source: DecisionSource
    expectation: float | None = None
    implication: str | None = None
    rule_family: str | None = None


def relation_fact(subject: str, value: str) -> str:
    """Build the binary property relation used by sorting observations."""
    return f"<({subject} * {value}) --> is>"


def operation_arguments(target: str) -> str:
    return f"({{SELF}} * {target})"


def color_action_target(action: str) -> str | None:
    prefix = 'place_to_bin_'
    if not action.startswith(prefix):
        return None
    color = action.removeprefix(prefix)
    if color and color[0] in 'abcdefghijklmnopqrstuvwxyz' and all(
        character in 'abcdefghijklmnopqrstuvwxyz0123456789' for character in color
    ):
        return color
    return None


def sorting_action_argument(action: str) -> str:
    """Convert a supported external action to its exact ONA target term."""
    prefix = 'place_to_bin'
    if action.startswith(prefix):
        number = action.removeprefix(prefix)
        if number and number[0] in '123456789' and all(c in '0123456789' for c in number):
            return f'bin{number}'
    color = color_action_target(action)
    if color is not None:
        return f'(bin * {color})'
    raise ValueError(f'Unsupported sorting action: {action!r}')


# Learned rules contain variables and nested terms; restrict pattern matching to
# diagnostic extraction instead of using it to construct known task symbols.
RELATIONAL_ROLE_PATTERN = re.compile(r"\((sphere|L1|L2) \* [^()]+\) --> is\b")


def classify_relational_rule(rule: str) -> str:
    """Classify an operation implication by its relational input factors."""

    roles = set(RELATIONAL_ROLE_PATTERN.findall(rule))
    factors = [
        label
        for role, label in (("sphere", "color"), ("L1", "L1"), ("L2", "L2"))
        if role in roles
    ]
    family = "+".join(factors)
    return family if family in RULE_FAMILIES else "unclassified"


class _ONAProcess:
    """Own shell I/O and lifetime; commands are opaque to the transport."""

    def __init__(self, binary_path: Path, seed: int, timeout: float) -> None:
        self.binary_path = binary_path
        self.seed = seed
        self.timeout = timeout
        self._output: Queue[str | None] = Queue()
        self._closed = False

        self._process = subprocess.Popen(
            [str(self.binary_path), "shell", "--seed", str(self.seed)],
            cwd=self.binary_path.parent,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        assert self._process.stdout is not None
        self._reader = Thread(
            target=self._read_stdout,
            args=(self._process.stdout,),
            daemon=True,
        )
        self._reader.start()

    @property
    def is_running(self) -> bool:
        return not self._closed and self._process.poll() is None

    def send_command(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> list[str]:
        """Send one shell command and return all output before its marker."""

        if not command or "\n" in command:
            raise ValueError("ONA command must be one non-empty line.")
        if not self.is_running:
            raise RuntimeError("ONA process is not running.")
        command_timeout = self.timeout if timeout is None else timeout
        if command_timeout <= 0:
            raise ValueError("ONA command timeout must be positive.")

        stdin = self._process.stdin
        assert stdin is not None
        stdin.write(f"{command}\n0\n")
        stdin.flush()

        lines: list[str] = []
        deadline = monotonic() + command_timeout
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError(f"ONA timed out while processing {command!r}.")
            try:
                line = self._output.get(timeout=remaining)
            except Empty as exc:
                raise TimeoutError(
                    f"ONA timed out while processing {command!r}."
                ) from exc
            if line is None:
                raise RuntimeError(
                    f"ONA exited with code {self._process.poll()} while processing "
                    f"{command!r}."
                )
            if line == RESPONSE_END:
                return lines
            lines.append(line)

    def close(self) -> None:
        """Stop the shell process. Calling close more than once is safe."""

        if self._closed:
            return
        self._closed = True

        if self._process.poll() is None:
            stdin = self._process.stdin
            if stdin is not None:
                try:
                    stdin.write("quit\n")
                    stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()

        self._reader.join(timeout=1.0)
        if self._process.stdin is not None:
            self._process.stdin.close()
        if self._process.stdout is not None:
            self._process.stdout.close()

    def _read_stdout(self, stdout: TextIO) -> None:
        for line in stdout:
            self._output.put(line.rstrip("\r\n"))
        self._output.put(None)


class ONAAgent(BaseAgent):
    """Use one long-running ``NAR shell`` process for an experiment run."""

    def __init__(
        self,
        action_space: Sequence[Action],
        seed: int,
        binary_path: str | Path = DEFAULT_ONA_BINARY,
        timeout: float = DEFAULT_ONA_TIMEOUT,
        startup_timeout: float = DEFAULT_ONA_STARTUP_TIMEOUT,
        inference_cycles: int = 100,
        anticipation_confidence: float | None = None,
        decision_threshold: float | None = None,
        interaction_encoding: InteractionEncoding = FLAT_ENCODING,
        name: str = "flat_ona",
    ) -> None:
        super().__init__(action_space, name=name)
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("ONA seed must be an unsigned 32-bit integer.")
        if timeout <= 0:
            raise ValueError("ONA timeout must be positive.")
        if startup_timeout <= 0:
            raise ValueError("ONA startup timeout must be positive.")
        if inference_cycles < 0:
            raise ValueError("ONA inference cycles must not be negative.")
        if anticipation_confidence is not None and not 0 <= anticipation_confidence <= 1:
            raise ValueError("ONA anticipation confidence must be between 0 and 1.")
        if decision_threshold is not None and not 0 <= decision_threshold <= 1:
            raise ValueError("ONA decision threshold must be between 0 and 1.")
        if interaction_encoding not in {FLAT_ENCODING, RELATIONAL_SORTING_ENCODING}:
            raise ValueError(
                f"Unknown ONA interaction encoding: {interaction_encoding!r}."
            )
        self.binary_path = Path(binary_path).resolve()
        if not self.binary_path.is_file():
            raise FileNotFoundError(
                f"ONA binary not found at {self.binary_path}. Build it with "
                "`python -m adaptive_sorting.experiments.ona_profile`."
            )

        self.seed = seed
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.inference_cycles = inference_cycles
        self.anticipation_confidence = anticipation_confidence
        self.decision_threshold = decision_threshold
        self.interaction_encoding = interaction_encoding
        self._set_operation_arguments(self._build_operation_argument_map())
        self._last_decision_diagnostic: DecisionDiagnostic | None = None
        self._process = _ONAProcess(self.binary_path, self.seed, self.timeout)
        try:
            self._configure_shell()
        except Exception:
            self.close()
            raise

    @property
    def is_running(self) -> bool:
        return self._process.is_running

    def send_command(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> list[str]:
        """Delegate shell I/O without interpreting the command or response."""
        return self._process.send_command(
            command, timeout=self.timeout if timeout is None else timeout,
        )

    def select_action(self, observation: Observation) -> Action:
        """Ask ONA for one operation for the current observation."""

        for index, term in enumerate(self._observation_terms(observation)):
            if index > 0:
                self.send_command("*concurrent")
            self.send_command(f"{term}. :|:")
        output = self.send_command(f"{GOAL}! :|:")
        action = self._parse_action(output)
        if action is not None:
            self._last_decision_diagnostic = self._decision_diagnostic(output)
            return action

        raise RuntimeError("ONA produced no operation for the current goal.")

    @property
    def last_decision_diagnostic(self) -> DecisionDiagnostic | None:
        return self._last_decision_diagnostic

    def _decision_diagnostic(self, output: Sequence[str]) -> DecisionDiagnostic:
        matches = [match for line in output if (match := DECISION_PATTERN.match(line))]
        if not matches:
            if any(line.startswith("decision expectation=") for line in output):
                return DecisionDiagnostic(source="unknown")
            return DecisionDiagnostic(source="motor_babbling")

        match = matches[-1]
        expectation = float(match.group("expectation"))
        decision_details = match.group("implication")
        implication = decision_details.split(". Stamp=[", maxsplit=1)[0]
        # Motor babbling is forced for exploration. ONA only replaces it when the
        # learned candidate exceeds this strict compile-time threshold.
        if expectation <= MOTOR_BABBLING_SUPPRESSION_THRESHOLD:
            return DecisionDiagnostic(source="motor_babbling")
        family = (
            classify_relational_rule(implication)
            if self.interaction_encoding == RELATIONAL_SORTING_ENCODING
            else None
        )
        return DecisionDiagnostic(
            source="learned_rule",
            expectation=expectation,
            implication=implication,
            rule_family=family,
        )

    def update(self, observation: Observation, action: Action, reward: int) -> None:
        """Report the executed action and scalar feedback to ONA."""

        if action not in self.action_space:
            raise ValueError(f"Unknown ONA action {action!r}.")
        if reward not in (-1, 1):
            raise ValueError("ONA reward must be -1 or 1.")

        truth = POSITIVE_FEEDBACK_TRUTH if reward > 0 else NEGATIVE_FEEDBACK_TRUTH
        feedback = f"{GOAL}. :|: {truth}"
        self.send_command(feedback)
        self.send_command(str(self.inference_cycles))

    def retained_operation_rules(self) -> tuple[str, ...]:
        """Return retained sorting implications from the current ONA memory."""

        if self.interaction_encoding != RELATIONAL_SORTING_ENCODING:
            raise RuntimeError("Rule snapshots require relational sorting encoding.")
        return tuple(
            line
            for line in self.send_command("*concepts")
            if GOAL in line and f"^{PARAMETERIZED_OPERATION}" in line
        )

    def configure_color_addressed_bin_actions(
        self,
        bin_colors: Sequence[tuple[str, str]],
    ) -> None:
        """Ground ``^place`` by a visible color on the generic bin target role."""

        if self.interaction_encoding != RELATIONAL_SORTING_ENCODING:
            raise RuntimeError("Color-addressed bins require relational encoding.")
        color_by_bin = dict(bin_colors)
        if len(color_by_bin) != len(bin_colors):
            raise ValueError("Color-addressed bins require unique bin names.")

        argument_to_action: dict[str, Action] = {}
        for action in self.action_space:
            color = color_action_target(action)
            if color is None:
                raise ValueError("Color-addressed actions must use place_to_bin_COLOR")
            if color_by_bin.pop(f"bin_{color}", None) != color:
                raise ValueError(f"Missing or inconsistent target color: {action}")
            argument_to_action[f"(bin * {color})"] = action
        if color_by_bin:
            raise ValueError("Observable colors contain unknown bins")

        self._set_operation_arguments(argument_to_action)
        for index, argument_term in enumerate(argument_to_action, start=1):
            self.send_command(f"*setoparg 1 {index} {argument_term}")

    def close(self) -> None:
        """Release the owned shell process; repeated calls are safe."""
        self._process.close()

    def __enter__(self) -> "ONAAgent":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _configure_shell(self) -> None:
        # The first response includes ONA's memory-heavy process initialization.
        self.send_command("*volume=0", timeout=self.startup_timeout)
        if self.anticipation_confidence is not None:
            self.send_command(
                f"*anticipationconfidence={self.anticipation_confidence}"
            )
        if self.decision_threshold is not None:
            self.send_command(f"*decisionthreshold={self.decision_threshold}")
        self.send_command(f"*motorbabbling={MOTOR_BABBLING_CHANCE}")
        if self.interaction_encoding == RELATIONAL_SORTING_ENCODING:
            self.send_command("*babblingops=1")
            self.send_command(f"*setopname 1 ^{PARAMETERIZED_OPERATION}")
            for index, argument_term in enumerate(
                self._operation_argument_to_action, start=1
            ):
                self.send_command(f"*setoparg 1 {index} {argument_term}")
        else:
            self.send_command(f"*babblingops={len(self.action_space)}")
            for index, action in enumerate(self.action_space, start=1):
                self.send_command(f"*setopname {index} ^{action}")

    def _parse_action(self, output: Sequence[str]) -> Action | None:
        for line in output:
            if " executed with args" not in line:
                continue
            operator, arguments = line.split(" executed with args", maxsplit=1)
            if self.interaction_encoding == RELATIONAL_SORTING_ENCODING:
                if operator != f"^{PARAMETERIZED_OPERATION}":
                    continue
                action = self._execution_argument_to_action.get(arguments.strip())
                if action is not None:
                    return action
                continue
            action = operator.removeprefix("^")
            if action in self.action_space:
                return action
        return None

    def _build_operation_argument_map(self) -> dict[str, Action]:
        if self.interaction_encoding == FLAT_ENCODING:
            return {}

        return {sorting_action_argument(action): action for action in self.action_space}

    def _set_operation_arguments(self, arguments: dict[str, Action]) -> None:
        self._operation_argument_to_action = arguments
        self._execution_argument_to_action = {
            operation_arguments(target): action for target, action in arguments.items()
        }

    def _observation_terms(self, observation: Observation) -> tuple[str, ...]:
        if self.interaction_encoding == FLAT_ENCODING:
            return (self._state_atom(observation),)

        terms = [relation_fact("sphere", observation.sphere_color)]
        if observation.context is None:
            return tuple(terms)

        light_states = context_light_states(observation.context)
        if observation.context in {"light_off", "light_on"}:
            light_states = light_states[:1]
        terms.extend(
            relation_fact(f"L{index}", "on" if is_on else "off")
            for index, is_on in enumerate(light_states, start=1)
        )
        return tuple(terms)

    @staticmethod
    def _state_atom(observation: Observation) -> str:
        return observation.key
