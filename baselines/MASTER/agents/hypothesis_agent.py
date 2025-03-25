from openai import OpenAI
import json
import random

class HypothesisAgent:
    def __init__(self, openai_api_key, openai_base_url, parameter_space, research_goal, research_constr):
        self.client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
        self.parameter_space = parameter_space
        self.research_goal = research_goal
        self.research_constr = research_constr

    def suggest_params(self, parent_params):
        prompt = f"""
        You are an expert in nanohelix experiments. Your task is to suggest a new set of parameters for an experiment while ensuring that the proposed values stay within the specified ranges and aim to maximize the g-factor.

        ### **Instructions:**
        - Follow the research goal: {self.research_goal}
        - Follow the research constraints: {self.research_constr}
        - The current parameter set is: {parent_params}
        - The allowed parameter space is: {self.parameter_space}
        - You must **ONLY** return a JSON object. **DO NOT** include any explanation, formatting, or text before or after the JSON.
        - The output must be a **valid JSON object** (no markdown, no additional text, no explanation).

        ### **Example Output Format (Strictly Follow This)**
        ```json
        {{
          "params": {{
            "pitch": 120.5,
            "curl": 2.8,
            "n_turns": 7,
            "fiber_radius": 35.0,
            "angle": 0.9,
            "height": 450.2,
            "helix_radius": 75.0,
            "total_fiber_length": 900.0,
            "total_length": 500.0
          }},
          "reasoning": "Increasing curl and helix_radius enhances g-factor by optimizing helical symmetry."
        }}
        ```
        Now, return a new parameter set in this exact format.
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        try:
            raw_content = response.choices[0].message.content

            if not raw_content.strip():
                raise ValueError("⚠️ Empty API response.")

            if raw_content.startswith("```json"):
                raw_content = raw_content[7:-3].strip()
            data = json.loads(raw_content)

            new_params = data.get("params", {})
            reasoning = data.get("reasoning", "")

            if not new_params:
                raise ValueError("⚠️ No 'params' key found in response.")

        except (json.JSONDecodeError, ValueError) as e:
            print(f"🚨 JSON Parsing Error: {e}")
            new_params = self.random_params()
            reasoning = "Fallback random guess"

        print("✅ **Parsed JSON Successfully:**", new_params, reasoning)
        return new_params, reasoning

    def random_params(self):
        ret = {}
        for var, (low, high) in self.parameter_space.items():
            val = random.uniform(low, high)
            if var == "n_turns":
                val = int(val)
            ret[var] = val
        return ret
