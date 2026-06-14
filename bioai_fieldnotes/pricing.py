from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .settings import env


@dataclass(frozen=True)
class ModelPrice:
    provider: str
    model: str
    input_per_million_usd: float
    output_per_million_usd: float
    source: str

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        return round(
            (input_tokens / 1_000_000) * self.input_per_million_usd
            + (output_tokens / 1_000_000) * self.output_per_million_usd,
            6,
        )


class PriceCatalog:
    def __init__(self, prices: list[ModelPrice]) -> None:
        self.prices = prices

    def get(self, provider: str, model: str) -> ModelPrice | None:
        provider = provider.lower()
        model = model.lower()
        for price in self.prices:
            if price.provider.lower() == provider and price.model.lower() == model:
                return price
        return None


def load_price_catalog(allow_fallback: bool = False) -> PriceCatalog:
    prices: list[ModelPrice] = []
    path = env("BIOAI_MODELPRICES_PATH")
    if path:
        prices.extend(_load_path(Path(path), source="modelprices"))
    if not prices and (allow_fallback or env("BIOAI_ALLOW_FALLBACK_PRICING") == "1"):
        fallback = Path(__file__).resolve().parent.parent / "configs" / "fallback_model_prices.json"
        prices.extend(_load_path(fallback, source="fallback"))
    return PriceCatalog(prices)


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def _load_path(path: Path, source: str) -> list[ModelPrice]:
    if path.is_dir():
        prices: list[ModelPrice] = []
        for child in sorted(path.iterdir()):
            if child.suffix.lower() in {".json", ".csv"}:
                prices.extend(_load_path(child, source=source))
        return prices
    if not path.exists():
        return []
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        records = payload.get("models", payload if isinstance(payload, list) else [])
        return [_coerce_price(record, source) for record in records if isinstance(record, dict)]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as fh:
            return [_coerce_price(row, source) for row in csv.DictReader(fh)]
    return []


def _coerce_price(record: dict[str, Any], source: str) -> ModelPrice:
    provider = str(record.get("provider") or record.get("vendor") or "openai")
    model = str(record.get("model") or record.get("name"))
    input_price = _float(
        record.get("input_per_million_usd")
        or record.get("input_price_per_million")
        or record.get("input")
        or record.get("prompt")
        or 0
    )
    output_price = _float(
        record.get("output_per_million_usd")
        or record.get("output_price_per_million")
        or record.get("output")
        or record.get("completion")
        or 0
    )
    return ModelPrice(provider, model, input_price, output_price, source)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
