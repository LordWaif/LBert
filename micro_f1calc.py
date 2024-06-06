import json

data = json.load(open("history.json"))
data = [v for _, v in data["evaluate"].items() if _ in list(map(str, range(0, 13)))]
# Dados extraídos da imagem
# data = [
#     {"class": 0, "precision": 0.70, "recall": 0.94, "f1-score": 0.80, "support": 360},
#     {"class": 1, "precision": 0.59, "recall": 0.75, "f1-score": 0.66, "support": 218},
#     {"class": 2, "precision": 0.91, "recall": 0.64, "f1-score": 0.75, "support": 108},
#     {"class": 3, "precision": 0.00, "recall": 0.00, "f1-score": 0.00, "support": 70},
#     {"class": 4, "precision": 0.00, "recall": 0.00, "f1-score": 0.00, "support": 22},
#     {"class": 5, "precision": 0.00, "recall": 0.00, "f1-score": 0.00, "support": 35},
#     {"class": 6, "precision": 0.73, "recall": 0.63, "f1-score": 0.67, "support": 51},
#     {"class": 7, "precision": 0.68, "recall": 0.69, "f1-score": 0.69, "support": 226},
#     {"class": 8, "precision": 0.35, "recall": 0.24, "f1-score": 0.28, "support": 165},
#     {"class": 9, "precision": 0.42, "recall": 0.06, "f1-score": 0.11, "support": 83},
#     {"class": 10, "precision": 0.00, "recall": 0.00, "f1-score": 0.00, "support": 134},
#     {"class": 11, "precision": 0.44, "recall": 0.95, "f1-score": 0.61, "support": 38},
#     {"class": 12, "precision": 0.00, "recall": 0.00, "f1-score": 0.00, "support": 10},
# ]

# Calculando TP, FP e FN
TP = sum([d["recall"] * d["support"] for d in data])
FP = sum([d["support"] * (1 - d["precision"]) for d in data if d["precision"] > 0])
FN = sum([d["support"] * (1 - d["recall"]) for d in data])

# Calculando micro-precision, micro-recall e micro-F1
micro_precision = TP / (TP + FP)
micro_recall = TP / (TP + FN)
micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

print(micro_precision, micro_recall, micro_f1)
# micro_precision, micro_recall, micro_f1
