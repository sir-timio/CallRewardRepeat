# https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/spokenwoz/LLM/dst/dst_data.py

domain_slot_description_dict = dict({
'hotel-pricerange': 'price budget of the hotel; Possible values: [\'expensive\', \'cheap\', \'moderate\'] ',
'hotel-type': 'type of the hotel; Possible values: [\'guest house\', \'hotel\']',
'hotel-parking': 'whether the hotel has parking; Possible values: [\'no\', \'yes\']',
'hotel-day': 'day of the hotel booking; Possible values: [\'monday\', \'tuesday\', \'wednesday\', \'thursday\', \'friday\', \'saturday\', \'sunday\']',
'hotel-people': 'number of people booking the hotel; Possible values: [\'1\', \'2\', \'3\', \'4\', \'5\', \'6\', \'7\', \'8\']',
'hotel-stay': 'length of stay at the hotel; Possible values: [\'1\', \'2\', \'3\', \'4\', \'5\', \'6\', \'7\', \'8\']',
'hotel-internet': 'whether the hotel has the free internet; Possible values: [\'no\', \'yes\']',
'hotel-name': 'name of the hotel; Possible values: []',
'hotel-area': 'area of the hotel; Possible values: [\'centre\', \'east\', \'north\', \'south\', \'west\']',
'hotel-star': 'star of the hotel; Possible values: [\'0\', \'1\', \'2\', \'3\', \'4\', \'5\']',
'train-arriveby': 'the arrival time of the train, 24-hour standard time, e.g. 06:00, 18:30; Possible values: []',
'train-day': 'day of the train departure; Possible values: [\'monday\', \'tuesday\', \'wednesday\', \'thursday\', \'friday\', \'saturday\', \'sunday\']',
'train-people': 'number of people travelling by train; Possible values: [\'1\', \'2\', \'3\', \'4\', \'5\', \'6\', \'7\', \'8\']',
'train-leaveat': 'leaving time of the train, 24-hour standard time, e.g. 06:00, 18:30; Possible values: []',
'train-destination': 'destination of the train; Possible values: [\'birmingham new street\', \'bishops stortford\', \'broxbourne\', \'cambridge\', \'ely\', \'kings lynn\', \'leicester\', \'london kings cross\', \'london liverpool street\', \'norwich\', \'peterborough\', \'stansted airport\', \'stevenage\']',
'train-departure': 'departure of the train; Possible values: [\'birmingham new street\', \'bishops stortford\', \'broxbourne\', \'cambridge\', \'ely\', \'kings lynn\', \'leicester\', \'london kings cross\', \'london liverpool street\', \'norwich\', \'peterborough\', \'stansted airport\', \'stevenage\']',
'attraction-area': 'area of the attraction; Possible values: [\'centre\', \'east\', \'north\', \'south\', \'west\']',
'attraction-name': 'name of the attraction; Possible values: []',
'attraction-type': 'type of the attraction; Possible values: [\'architecture\', \'boat\', \'cinema\', \'college\', \'concerthall\', \'entertainment\', \'museum\', \'multiple sports\', \'nightclub\', \'park\', \'swimmingpool\', \'theatre\']',
'restaurant-pricerange': 'price budget for the restaurant; Possible values: [\'expensive\', \'cheap\', \'moderate\']',
'restaurant-area': 'area of the restaurant; Possible values: [\'centre\', \'east\', \'north\', \'south\', \'west\']',
'restaurant-food': 'the cuisine of the restaurant; Possible values: []',
'restaurant-name': 'name of the restaurant; Possible values: []',
'restaurant-day': 'day of the restaurant booking; Possible values: [\'monday\', \'tuesday\', \'wednesday\', \'thursday\', \'friday\', \'saturday\', \'sunday\']',
'restaurant-people': 'number of people for the restaurant booking; Possible values: [\'1\', \'2\', \'3\', \'4\', \'5\', \'6\', \'7\', \'8\']',
'restaurant-time': 'time of the restaurant booking, 24-hour standard time, e.g. 06:00, 18:30; Possible values: []',
'hospital-department': 'department of the hospital; Possible values: []',
'taxi-leaveat': 'leaving time of taxi, 24-hour standard time, e.g. 06:00, 18:30; Possible values: []',
'taxi-destination': "destination of taxi; Possible values: []",
"taxi-departure": "departure location of taxi; Possible values: []",
"taxi-arriveby": "arrival time of taxi, 24-hour standard time, e.g. 06:00, 18:30; Possible values: []",
"profile-name": 'the name of the user; Possible values: []',
"profile-email": 'the email of the user; Possible values: []',
"profile-idnumber": 'the idnumber of the user; Possible values: []',
"profile-phonenumber": 'the phonenumber of the user; Possible values: []',
"profile-platenumber": 'the platenumber of the user; Possible values: []',
})

# canonical “booking” slots for each domain (MultiWOZ convention)
# BOOK_SLOTS = {
#     "hotel":      ["day", "people", "stay"],
#     "restaurant": ["day", "people", "time"],
#     "train":      ["day", "departure", "destination", "leave_at_or_after"],
#     "taxi":       ["departure", "destination", "leave_at_or_after", "arrive_by"],
# }
# generic action descriptions (UBAR / FnCTOD)
ACT_DESC = {
    "inform":      "provide information about an entity (required)",
    "request":     "ask the user for more preference to narrow down the search (optional)",
    "recommend":   "recommend an offer (optional)",
    "select":      "ask the user to choose among available offers (optional)",
    "offerbook":   "offer an entity and ask if the user would like to book it (required)",
    "offerbooked": "confirm a successful booking and provide reference details (required)",
    "nooffer":     "inform the user that no suitable offer could be found",
    "nobook":      "inform the user that the offer cannot be booked",
    "general":     "greet / reqmore / bye"
}


# hotel
# train
# attraction
# restaurant
# hospital
# taxi
# profile
FIND_DOMAINS = {"hotel", "train", "attraction", "restaurant", "hospital"}
BOOK_DOMAINS = {"taxi"}
FILL_DOMAINS = {"profile"}

               
import json
import re
import argparse
import yaml
from pathlib import Path

def split_domain_slot(key: str):
    """'hotel-pricerange' -> ('hotel', 'pricerange')"""
    head, tail = key.split("-", 1)
    return head, tail

def is_categorical(values):
    return bool(values)

def parse_possible_values(desc: str):
    """
    "Possible values: ['expensive','cheap']" -> ['expensive','cheap']
    """
    m = re.search(r"\[([^\]]*)\]", desc)
    if not m:
        return []
    raw = m.group(1)
    return [v.strip(" '\"") for v in raw.split(",") if v.strip()]

def build_schema(domain_slot_description: dict) -> list[dict]:

    services = {}
    for full_slot, description in domain_slot_description.items():
        domain, _ = split_domain_slot(full_slot)
        pv = parse_possible_values(description)
        services.setdefault(domain, {
            "service_name": domain,
            "description": f"{domain} domain",
            "slots": [],
            "intents": [],
            "actions": []
        })
        services[domain]["slots"].append({
            "name": full_slot,
            "description": re.sub(r";?\s*Possible values:.*", "", description).strip(),
            "possible_values": pv,
            "is_categorical": is_categorical(pv),
            "is_informable": True,
            "is_requestable": True
        })

    # add intents & actions ----------------------------------------------------
    for dom, svc in services.items():
        all_slots = [s["name"].split("-",1)[1] for s in svc["slots"]]
        optional  = {s: "dontcare" for s in all_slots}

        if dom in FIND_DOMAINS:   # FIND intent
            svc["intents"].append({
                "name":            f"find_{dom}",
                "description":     f"search for a {dom}",
                "is_transactional": False,
                "required_slots":   [],
                "optional_slots":   optional
            })
        if dom in BOOK_DOMAINS:   # BOOK intent
            svc["intents"].append({
                "name":            f"book_{dom}",
                "description":     f"book a {dom}",
                "is_transactional": True,
                "required_slots":   [],
                "optional_slots":   optional
            })
        if dom in FILL_DOMAINS:   # BOOK intent
            svc["intents"].append({
                "name":            f"fill_{dom}",
                "description":     f"fill a {dom}",
                "is_transactional": False,
                "required_slots":   [],
                "optional_slots":   optional
            })

        # add standard actions
        svc["actions"] = [{"name": n, "description": d} for n,d in ACT_DESC.items()]

    return list(services.values())

def main():
    parser = argparse.ArgumentParser(
        description="Normalize domain_slot_description_dict → FnCTOD schema YAML"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output YAML file"
    )
    args = parser.parse_args()

    schema = build_schema(domain_slot_description_dict)
    yaml_text = yaml.safe_dump(schema, sort_keys=False, allow_unicode=True)
    Path(args.output).write_text(yaml_text, encoding="utf-8")
    print(f"Saved normalized schema ({len(schema)} services) to {args.output}")

if __name__ == "__main__":
    main()