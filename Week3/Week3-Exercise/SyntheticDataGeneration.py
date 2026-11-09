
import json
import os
import math
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv


# ============================================================
# 1. CONFIGURATION
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Get the directory containing this Python file
BASE_DIR = Path(__file__).resolve().parent

# Locate .env in the same directory
ENV_PATH = BASE_DIR / ".env"

# Load environment variables
load_dotenv(dotenv_path=ENV_PATH, override=True)

# Retrieve API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Debug checks
print("Python file location:", __file__)
print("Environment file location:", ENV_PATH)
print("Does .env exist?", ENV_PATH.exists())
print("API key loaded?", bool(OPENROUTER_API_KEY))

if not OPENROUTER_API_KEY:
    raise ValueError(
        f"OPENROUTER_API_KEY not found. Check: {ENV_PATH}"
    )

MODEL = "nvidia/nemotron-3.5-lightning:free"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

print("OpenRouter client initialized successfully!")


# ============================================================
# 2. DATASET DEFINITIONS
# ============================================================

DEFINITIONS = {
    "valid": (
        "Allocations satisfy demand and do not exceed "
        "the capacity of any supplier."
    ),

    "demand_violation": (
        "Total allocation is less than demand, "
        "but no supplier allocation exceeds its capacity."
    ),

    "capacity_violation": (
        "At least one supplier allocation exceeds its capacity."
    ),

    "borderline": (
        "Allocations are exactly equal or very close "
        "to demand or supplier capacity limits, "
        "without violating those limits."
    ),
}


# ============================================================
# 3. EXAMPLE SAMPLES
# ============================================================

EXAMPLES = {
    "valid": {
        "type": "valid",
        "capacities": {
            "S1": 500,
            "S2": 300
        },
        "demand": 600,
        "allocations": {
            "S1": 350,
            "S2": 250
        }
    },

    "demand_violation": {
        "type": "demand_violation",
        "capacities": {
            "S1": 500,
            "S2": 300
        },
        "demand": 600,
        "allocations": {
            "S1": 200,
            "S2": 150
        }
    },

    "capacity_violation": {
        "type": "capacity_violation",
        "capacities": {
            "S1": 400,
            "S2": 300
        },
        "demand": 600,
        "allocations": {
            "S1": 450,
            "S2": 150
        }
    },

    "borderline": {
        "type": "borderline",
        "capacities": {
            "S1": 500,
            "S2": 300
        },
        "demand": 600,
        "allocations": {
            "S1": 500,
            "S2": 100
        }
    },
}


# ============================================================
# 4. BUILD MESSAGES
# ============================================================

def build_messages(dataset_type, n_samples):

    if dataset_type not in DEFINITIONS:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}"
        )

    definition = DEFINITIONS[dataset_type]

    example = json.dumps(
        EXAMPLES[dataset_type],
        indent=2
    )

    system_content = (
        "You are an expert synthetic data generator. "
        "Your task is to generate realistic and diverse "
        "numerical allocation datasets. "
        "You must output only valid JSON arrays. "
        "Do not include explanations, markdown, or code blocks. "
        "Every sample must contain exactly these keys: "
        "type, capacities, demand, allocations."
    )

    user_content = (
        f"Generate exactly {n_samples} samples "
        f"of type '{dataset_type}'.\n\n"

        f"Definition: {definition}\n\n"

        "Here is an example sample. "
        "Use it only as a structural reference. "
        "Do not copy its numerical values:\n\n"

        f"{example}\n\n"

        "Requirements:\n"
        "1. Every sample must have the requested type.\n"
        "2. Generate different numerical values across samples.\n"
        "3. Capacities and allocations must use matching supplier names.\n"
        "4. Demand and capacities must be positive numbers.\n"
        "5. Allocations must be non-negative numbers.\n"
        "6. Follow the requested dataset definition strictly.\n"
        "7. Return exactly the requested number of samples.\n"
        "8. Use at least two suppliers per sample.\n\n"

        "Return only a JSON array. "
        "Do not include any additional text."
    )

    return [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": user_content
        }
    ]


# ============================================================
# 5. PARSE JSON RESPONSE
# ============================================================

def parse_json_response(response_text):

    response_text = response_text.strip()

    # Remove markdown code fences if present
    if response_text.startswith("```"):

        lines = response_text.splitlines()

        if lines[-1].strip() == "```":
            lines = lines[:-1]

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        response_text = "\n".join(lines)

    # Locate the JSON array
    start = response_text.find("[")
    end = response_text.rfind("]")

    if start == -1 or end == -1:
        raise ValueError(
            "No JSON array found in the model response."
        )

    json_text = response_text[start:end + 1]

    try:
        return json.loads(json_text)

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON returned by the model: {e}"
        ) from e


# ============================================================
# 6. DATA VALIDATION
# ============================================================

def is_valid_number(value, allow_zero=False):

    if isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    if not math.isfinite(value):
        return False

    if allow_zero:
        return value >= 0

    return value > 0


def validate_sample(sample, dataset_type):

    required_keys = {
        "type",
        "capacities",
        "demand",
        "allocations"
    }

    # Check sample structure
    if not isinstance(sample, dict):
        return False

    if set(sample.keys()) != required_keys:
        return False

    # Check dataset type
    if sample["type"] != dataset_type:
        return False

    capacities = sample["capacities"]
    allocations = sample["allocations"]
    demand = sample["demand"]

    # Check dictionary structures
    if not isinstance(capacities, dict):
        return False

    if not isinstance(allocations, dict):
        return False

    if len(capacities) < 2:
        return False

    # Supplier names must match
    if set(capacities.keys()) != set(allocations.keys()):
        return False

    # Demand must be positive
    if not is_valid_number(demand):
        return False

    # Validate capacities and allocations
    for supplier in capacities:

        capacity = capacities[supplier]
        allocation = allocations[supplier]

        if not is_valid_number(capacity):
            return False

        if not is_valid_number(allocation, allow_zero=True):
            return False

    # Calculate total allocation
    total_allocation = sum(allocations.values())

    # Check capacity violations
    has_capacity_violation = any(
        allocations[supplier] > capacities[supplier]
        for supplier in capacities
    )

    # Check demand violation
    has_demand_violation = total_allocation < demand

    # --------------------------------------------------------
    # VALID
    # --------------------------------------------------------

    if dataset_type == "valid":

        return (
            total_allocation >= demand
            and not has_capacity_violation
        )

    # --------------------------------------------------------
    # DEMAND VIOLATION
    # --------------------------------------------------------

    elif dataset_type == "demand_violation":

        return (
            has_demand_violation
            and not has_capacity_violation
        )

    # --------------------------------------------------------
    # CAPACITY VIOLATION
    # --------------------------------------------------------

    elif dataset_type == "capacity_violation":

        return has_capacity_violation

    # --------------------------------------------------------
    # BORDERLINE
    # --------------------------------------------------------

    elif dataset_type == "borderline":

        # Borderline samples must not violate constraints
        if has_capacity_violation or has_demand_violation:
            return False

        tolerance = 0.05

        # Check whether any allocation is close to capacity
        close_to_capacity = any(
            abs(capacities[s] - allocations[s])
            <= capacities[s] * tolerance
            for s in capacities
        )

        # Check whether total allocation is close to demand
        close_to_demand = (
            abs(total_allocation - demand)
            <= demand * tolerance
        )

        return close_to_capacity or close_to_demand

    return False


# ============================================================
# 7. GENERATE DATASET USING NVIDIA NEMOTRON
# ============================================================

def generate_dataset(dataset_type, n_samples):

    if dataset_type not in DEFINITIONS:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}"
        )

    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples <= 0:
        raise ValueError(
            "n_samples must be a positive integer."
        )

    print(f"\nGenerating {n_samples} samples...")
    print(f"Dataset type: {dataset_type}")
    print(f"Model: {MODEL}\n")

    messages = build_messages(
        dataset_type,
        n_samples
    )

    # Send request to OpenRouter
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=4096,
        temperature=0.8
    )

    response_text = response.choices[0].message.content

    if not response_text:
        raise ValueError(
            "The model returned an empty response."
        )

    # Parse JSON
    dataset = parse_json_response(response_text)

    if not isinstance(dataset, list):
        raise ValueError(
            "Expected a JSON array from the model."
        )

    # Check number of samples
    if len(dataset) != n_samples:
        raise ValueError(
            f"Expected {n_samples} samples, "
            f"but received {len(dataset)}."
        )

    # Validate every sample
    invalid_samples = []

    for index, sample in enumerate(dataset):

        if not validate_sample(sample, dataset_type):
            invalid_samples.append(index + 1)

    if invalid_samples:

        raise ValueError(
            f"Invalid samples detected at positions: "
            f"{invalid_samples}"
        )

    print("All samples passed validation!")

    return dataset


# ============================================================
# 8. SAVE DATASET
# ============================================================

def save_dataset(dataset, dataset_type):

    output_file = (
        Path(__file__).resolve().parent
        / f"{dataset_type}_dataset.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dataset,
            file,
            indent=2
        )

    print(f"\nDataset saved to: {output_file}")

    return output_file


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    dataset_type = "valid"
    n_samples = 5

    try:

        dataset = generate_dataset(
            dataset_type,
            n_samples
        )

        print("\nGenerated Dataset:\n")

        print(
            json.dumps(
                dataset,
                indent=2
            )
        )

        save_dataset(
            dataset,
            dataset_type
        )

    except Exception as e:

        print(f"\nDataset generation failed: {e}")
