"""Shared source of truth for the llguidance POC.

Ten tools, each exercising a "JSON but more rigorous" value domain. Every value is
emitted as a quoted JSON string (matching Needle's observed output format), so the
grammar constrains the *string content* per domain. The same specs drive:
  - the tool JSON shown to the model (encoder input, all 10 tools always),
  - the union Lark grammar (llguidance), and
  - the per-domain validators (benchmark scoring).
"""
import json
import re

# Each tool: ordered list of (key, lark_regex, human_desc). lark_regex is the byte
# pattern the *string content* of that value must match. Validators below re-check
# semantics the grammar can't fully express (real calendar dates, from!=to, ranges).

US_STATES = ["AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA",
    "WA","WV","WI","WY"]
CURRENCIES = ["USD","EUR","GBP","JPY","CAD","AUD","CHF","CNY","INR","MXN"]

TOOLS = {
    "schedule_event": {
        "desc": "Schedule a calendar event at a specific date and time.",
        "args": [
            ("date", r"20[0-9][0-9]-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])",
             "Date in strict YYYY-MM-DD format."),
            ("time", r"([01][0-9]|2[0-3]):[0-5][0-9]",
             "24-hour time in strict HH:MM format."),
        ],
    },
    "convert_currency": {
        "desc": "Convert an amount of money between two ISO-4217 currencies.",
        "args": [
            ("amount", r"[0-9]{1,6}(\.[0-9]{1,2})?", "Decimal amount, no symbols."),
            ("from_code", "|".join(CURRENCIES), "3-letter source currency code."),
            ("to_code", "|".join(CURRENCIES), "3-letter target currency code."),
        ],
    },
    "set_shipping_address": {
        "desc": "Set the shipping destination by US state and ZIP code.",
        "args": [
            ("state", "|".join(US_STATES), "2-letter US state postal code."),
            ("zip", r"[0-9]{5}", "5-digit US ZIP code."),
        ],
    },
    "rate_product": {
        "desc": "Rate a product with a star rating from 1 to 5.",
        "args": [("stars", r"[1-5]", "Integer 1-5.")],
    },
    "set_brightness": {
        "desc": "Set the display brightness as a percentage from 0 to 100.",
        "args": [("percent", r"100|[1-9][0-9]|[0-9]", "Integer 0-100.")],
    },
    "dial_phone": {
        "desc": "Dial a US phone number.",
        "args": [("number", r"[2-9][0-9]{2}-[0-9]{3}-[0-9]{4}",
                  "US phone number NNN-NNN-NNNN.")],
    },
    "set_color": {
        "desc": "Set an RGB color by hex code.",
        "args": [("hex", r"#[0-9A-Fa-f]{6}", "Hex color #RRGGBB.")],
    },
    "set_thermostat": {
        "desc": "Set the thermostat temperature and unit.",
        "args": [
            ("temperature", r"[5-8][0-9]|90", "Integer 50-90."),
            ("unit", r"F|C", "Temperature unit F or C."),
        ],
    },
    "set_waypoint": {
        "desc": "Set a navigation waypoint by latitude and longitude.",
        "args": [
            ("lat", r"-?(90(\.0+)?|[0-8]?[0-9](\.[0-9]{1,6})?)",
             "Latitude decimal degrees -90..90."),
            ("lon", r"-?(180(\.0+)?|1[0-7][0-9](\.[0-9]{1,6})?|[0-9]?[0-9](\.[0-9]{1,6})?)",
             "Longitude decimal degrees -180..180."),
        ],
    },
    "set_timer": {
        "desc": "Set a countdown timer.",
        "args": [
            ("hours", r"2[0-3]|1[0-9]|[0-9]", "Integer hours 0-23."),
            ("minutes", r"[1-5][0-9]|[0-9]", "Integer minutes 0-59."),
            ("seconds", r"[1-5][0-9]|[0-9]", "Integer seconds 0-59."),
        ],
    },
}

DOMAINS = list(TOOLS.keys())


def tools_json_all():
    """The tool list shown to the model on every query (all 10 tools)."""
    out = []
    for name, spec in TOOLS.items():
        params = {}
        for key, _rx, desc in spec["args"]:
            params[key] = {"type": "string", "description": desc, "required": True}
        out.append({"name": name, "description": spec["desc"], "parameters": params})
    return json.dumps(out, separators=(",", ":"))


def _lit(s):
    """Return a Lark double-quoted string literal that matches the bytes of *s*."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_union_grammar():
    """Union Lark grammar over all 10 tool calls.

    Shared forced prefix ' [{"name":"' (matches Needle's leading-space token), then
    the tool name is the one real decision, then forced structure with regex value
    slots. Envelope: ' [{"name":"TOOL","arguments":{"k1":"v1",...}}]'.
    """
    lines = []
    alts = [f"t_{name}" for name in TOOLS]
    lines.append("start: " + _lit(' [{"name":"') + " body")
    lines.append("body: " + " | ".join(alts))
    term_defs = []
    for name, spec in TOOLS.items():
        # after the shared '"name":"' prefix: TOOLNAME","arguments":{
        parts = [_lit(name + '","arguments":{')]
        for i, (key, rx, _desc) in enumerate(spec["args"]):
            sep = "" if i == 0 else ","
            term = ("R_" + name + "_" + key).upper()
            term_defs.append(f"{term}: /{rx}/")
            parts.append(_lit(sep + '"' + key + '":"') + " " + term + " " + _lit('"'))
        parts.append(_lit("}}]"))  # close arguments + object + array
        lines.append(f"t_{name}: " + " ".join(parts))
    lines.extend(term_defs)
    return "\n".join(lines)


# ---------------- validators ----------------

def _valid_date(s):
    m = re.fullmatch(r"(20[0-9][0-9])-(\d{2})-(\d{2})", s)
    if not m:
        return False
    y, mo, d = int(m[1]), int(m[2]), int(m[3])
    if not (1 <= mo <= 12):
        return False
    days = [31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28,
            31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return 1 <= d <= days[mo - 1]


def _value_valid(domain, key, val):
    """Semantic validity of a single value (stricter than the regex where needed)."""
    if not isinstance(val, str):
        val = str(val)
    if domain == "schedule_event" and key == "date":
        return _valid_date(val)
    if domain == "schedule_event" and key == "time":
        return re.fullmatch(r"([01][0-9]|2[0-3]):[0-5][0-9]", val) is not None
    if domain == "convert_currency":
        if key == "amount":
            return re.fullmatch(r"[0-9]{1,6}(\.[0-9]{1,2})?", val) is not None
        return val in CURRENCIES
    if domain == "set_shipping_address":
        if key == "state":
            return val in US_STATES
        return re.fullmatch(r"[0-9]{5}", val) is not None
    if domain == "rate_product":
        return val in {"1", "2", "3", "4", "5"}
    if domain == "set_brightness":
        return val.isdigit() and 0 <= int(val) <= 100 and (val == "0" or val[0] != "0")
    if domain == "dial_phone":
        return re.fullmatch(r"[2-9][0-9]{2}-[0-9]{3}-[0-9]{4}", val) is not None
    if domain == "set_color":
        return re.fullmatch(r"#[0-9A-Fa-f]{6}", val) is not None
    if domain == "set_thermostat":
        if key == "temperature":
            return val.isdigit() and 50 <= int(val) <= 90
        return val in {"F", "C"}
    if domain == "set_waypoint":
        try:
            f = float(val)
        except ValueError:
            return False
        return (-90 <= f <= 90) if key == "lat" else (-180 <= f <= 180)
    if domain == "set_timer":
        if not val.isdigit():
            return False
        n = int(val)
        return 0 <= n <= (23 if key == "hours" else 59)
    return False


def score_call(domain, parsed):
    """Score a parsed tool call (list or dict) against the expected domain.

    Returns dict with booleans: json_ok, name_ok, keys_ok, values_ok.
    values_ok requires name_ok AND keys_ok AND every value semantically valid.
    """
    res = {"json_ok": False, "name_ok": False, "keys_ok": False, "values_ok": False}
    if parsed is None:
        return res
    res["json_ok"] = True
    call = parsed[0] if isinstance(parsed, list) and parsed else parsed
    if not isinstance(call, dict):
        return res
    name = call.get("name")
    res["name_ok"] = (name == domain)
    args = call.get("arguments", {})
    if not isinstance(args, dict):
        return res
    expected_keys = [k for k, _r, _d in TOOLS[domain]["args"]]
    res["keys_ok"] = res["name_ok"] and set(args.keys()) == set(expected_keys)
    if res["keys_ok"]:
        res["values_ok"] = all(_value_valid(domain, k, args.get(k)) for k in expected_keys)
    return res


def exact_match(domain, parsed, expected_args):
    call = parsed[0] if isinstance(parsed, list) and parsed else parsed
    if not isinstance(call, dict) or call.get("name") != domain:
        return False
    args = call.get("arguments", {})
    if not isinstance(args, dict):
        return False
    exp = {k: str(v) for k, v in expected_args.items()}
    got = {k: str(v) for k, v in args.items()}
    if domain == "set_color":  # case-insensitive hex compare
        exp = {k: v.upper() for k, v in exp.items()}
        got = {k: v.upper() for k, v in got.items()}
    return got == exp


if __name__ == "__main__":
    g = build_union_grammar()
    print(g)
    print("\n--- tools_json (first 400 chars) ---")
    print(tools_json_all()[:400])
