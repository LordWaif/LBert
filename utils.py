import json


def collect_labels(labels, sub_task):
    if sub_task == "multi_label":
        labels = [item for sublist in labels for item in sublist]
    elif sub_task == "multi_class":
        labels = labels
    lb = set(labels)
    labels_map = {_: _i for _i, _ in enumerate(lb)}
    json.dump(labels_map, open("labels.json", "w"), ensure_ascii=False, indent=4)
