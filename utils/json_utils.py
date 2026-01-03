import json
import json5

def str_to_json(input_str):
    if input_str.startswith("json"):
        input_str = input_str[len("json"):]
    if input_str.endswith("json"):
        input_str = input_str[:-len("json")]
    input_str = input_str.strip().replace("\n", "").replace("```json", "").replace("```", "")
    try:
        return json.loads(input_str, strict=False)
    except:
        pass
    try:
        return json5.loads(input_str)
    except:
        pass
    print(input_str)
    raise ValueError("Invalid JSON")
    