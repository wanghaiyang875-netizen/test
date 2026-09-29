import json
import re
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

def parse_model_output(value, model: Type[T]) -> T:
    text = str(getattr(value, "content", value)).strip()
    match = re.search(r"\{.*?\}", text, flags=re.S)
    for candidate in ([match.group(0)] if match else []) + [text]:
        try:
            return model.model_validate(json.loads(candidate))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    if model.__name__ == "RouteQuery":
        source = "web_search" if "web_search" in text.lower() or "网络" in text else "vectorstore"
        return model.model_validate({"datasource": source})
    score = "yes" if re.search(r"\byes\b|是|相关|支持", text.lower()) else "no"
    return model.model_validate({"binary_score": score})
