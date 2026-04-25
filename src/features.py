import math
import numpy as np


def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def shot_distance(x, y):
    goal_x, goal_y = 120, 40
    return math.hypot(goal_x - x, goal_y - y)


def shot_angle(x, y):
    if x > 60:
        left_post = (120, 36)
        right_post = (120, 44)
    else:
        left_post = (0, 36)
        right_post = (0, 44)

    a = (left_post[0] - x, left_post[1] - y)
    b = (right_post[0] - x, right_post[1] - y)

    dot = a[0]*b[0] + a[1]*b[1]
    mag_a = math.hypot(*a)
    mag_b = math.hypot(*b)
    angle = math.acos(dot / (mag_a * mag_b))
    return math.degrees(angle)


def safe_get_gk(freeze_frame):
    for p in freeze_frame:
        pos = p.get("position", {}).get("name", "").lower()
        if "goalkeeper" in pos:
            return p
    return None


def extract_freezefeature(event):
    ff = event.get("shot", {}).get("freeze_frame", None)

    features = {
        "n_def_1_5": 0,
        "n_def_3_0": 0,
        "dist_nearest_def": 99.0 / 127,
        "gk_dist_to_shooter": 11 / 127,
    }

    sx, sy = event.get("location", [np.nan, np.nan])

    if not ff:
        return features

    defender_distances = []
    for p in ff:
        if not p["teammate"]:
            loc = p["location"]
            d = euclid((sx, sy), (loc[0], loc[1]))
            defender_distances.append(d)

    if defender_distances:
        features["n_def_1_5"] = sum(1 for d in defender_distances if d <= 1.5)
        features["n_def_3_0"] = sum(1 for d in defender_distances if d <= 3.0)
        features["dist_nearest_def"] = float(np.min(defender_distances)) / 127

    gkp = safe_get_gk(ff)
    if gkp:
        gx, gy = gkp.get("location", [np.nan, np.nan])
        features["gk_dist_to_shooter"] = euclid((sx, sy), (gx, gy)) / 127

    return features


def find_event_by_id(events, event_id):
    for e in events:
        if e["id"] == event_id:
            return e
    return None

def extract_pass_features(event, events, pass_id):
    features = {
        "pass_player": None,
        "pass_start_distance": None,
        "pass_length": None,
        "pass_end_distance": None,
        "pass_angle": None,
        "pass_height": None,
        "pass_type": None,
        "cross": None,
        "cut_back": None,
        "pass_height_b": None,
        "pass_type_b": None
    }

    if pass_id is None:
        return features

    pass_event = find_event_by_id(events, pass_id)

    if not pass_event or pass_event["type"]["name"] != "Pass":
        return features

    passer_x, passer_y = pass_event["location"]
    p = pass_event.get("pass", {})

    length = p.get("length")
    angle = p.get("angle")
    pass_type = p.get("type", {}).get("name")

    features.update({
        "pass_player": pass_event["player"]["name"],
        "pass_start_distance": shot_distance(passer_x, passer_y) / 127,
        "pass_length": length / 144 if length is not None else None,
        "pass_angle": (angle + math.pi) / (2 * math.pi) if angle is not None else None,
        "pass_height": p.get("height", {}).get("name"),
        "pass_type": pass_type if pass_type is not None else "Open Play",
        "cross": int(p.get("cross", False)),
        "cut_back": int(p.get("cut_back", False)),
        "pass_height_b": 0 if p.get("height", {}).get("name") == "Ground Pass" else 1,
        "pass_type_b": 0 if pass_type is None else 1
    })

    if "end_location" in p:
        pass_x, pass_y = p["end_location"]
        features["pass_end_distance"] = shot_distance(pass_x, pass_y) / 127

    return features