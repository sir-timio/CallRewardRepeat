#!/usr/bin/env python3
# spokenwoz2fnctod.py
#
# Example:
#   python spokenwoz2fnctod.py \
#       --input ../data/spokenwoz/test/data.json \
#       --output spokenwoz_fnctod_dst.json

import json
import argparse
import sys
from pathlib import Path
from collections import OrderedDict
from src.multiwoz.postprocess import compare_dict


def normalize_slot(slot: str, domain: str) -> str:
    mapping = {
        "bookstay": "stay",
        "bookday": "day",
        "bookpeople": "people",
        "booktime": "time",
        "arriveby": "arrive_by",
        "leaveat": "leave_at_or_after",
        "ref": "reference",
    }
    if slot == "type" and domain == "hotel":
        return "accommodation_type"
    return mapping.get(slot, slot)


def meta2dict(meta: dict) -> dict:
    """meta → {'[domain]': {slot: value, …}, …}"""
    belief = {}
    for dom, dom_data in meta.items():
        slots = {}

        # semi
        for slot, val in dom_data.get("semi", {}).items():
            if val and (v := str(val).strip()):
                slot = normalize_slot(slot.lower(), dom)
                slots[slot] = v.lower()

        # book
        for slot, val in dom_data.get("book", {}).items():
            if slot == "booked":
                continue
            if val and (v := str(val).strip()):
                slot = normalize_slot(slot.lower(), dom)
                slots[slot] = v.lower()

        if slots:
            belief[f"[{dom}]"] = slots
    return belief


def dict2bs(belief: dict) -> str:
    tokens = []
    for dom in sorted(belief.keys()):
        tokens.append(dom)
        for slot in sorted(belief[dom].keys()):
            tokens.extend([slot, belief[dom][slot]])
    return " ".join(tokens)


def diff_dict(old: dict, new: dict) -> dict:
    diff = {}
    for dom in new:
        added = {
            s: v for s, v in new[dom].items() if dom not in old or old[dom].get(s) != v
        }
        if added:
            diff[dom] = added
    return diff


# ------------------------------------------------------------------ converter
def convert_dialogs(data):
    all_dialogs = {}
    for dial_id, sample in data.items():
        log = sample["log"]
        turns = []
        prev_state = OrderedDict()

        for i in range(0, len(log) - 1, 2):
            if log[i]["tag"] != "user" or log[i + 1]["tag"] != "system":
                print(f"[WARN] {dial_id} turn {i}: bad tag order", file=sys.stderr)
                continue

            user_txt = log[i]["text"]
            sys_txt = log[i + 1]["text"]
            curr_bs = meta2dict(log[i + 1]["metadata"])
            diff_bs = compare_dict(prev_state, curr_bs)

            if diff_bs:
                dspn = next(iter(diff_bs))
            else:
                dspn = "[general]"

            turn = {
                "dial_id": dial_id,
                "turn_num": i // 2,
                "user": user_txt,
                "resp": "",  # empty, not needed for DST
                "nodelx_resp": sys_txt,
                "bspn_dict": curr_bs,
                "bspn": dict2bs(curr_bs),
                "turn_bspn_dict": diff_bs,
                "turn_bspn": dict2bs(diff_bs),
                "bsdx": "",
                "db": 0,  # no real db
                "dspn": dspn,
                "aspn": "",
                "aspn_dict": {},
                "all_domains": [], # will fill in later
            }

            turns.append(turn)
            prev_state = curr_bs

        # all_domains (multiple domains per dialogue)
        doms = {t["dspn"] for t in turns if t["dspn"] != "[general]"}
        for t in turns:
            t["all_domains"] = list(doms)
        all_dialogs[dial_id] = turns
    return all_dialogs


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Convert SpokenWOZ to FnCTOD-style DST json"
    )
    p.add_argument("--input", required=True, help="path to spokenwoz data.json")
    p.add_argument("--output", required=True, help="where to write result")
    args = p.parse_args()

    data = json.load(open(args.input, "r", encoding="utf-8"))
    converted = convert_dialogs(data)

    Path(args.output).write_text(
        json.dumps(converted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(converted)} dialogues → {args.output}")


if __name__ == "__main__":
    main()
