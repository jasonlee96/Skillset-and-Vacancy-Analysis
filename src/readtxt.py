from unicodecsv import csv

def get_vocab():
    with open('input/vocab.txt', encoding='utf-8') as source:
        rdr = csv.reader(source)
        features = []
        skill_set = set()
        for row in rdr:
            skill = row[0].lower()
            if skill not in skill_set:
                skill_set.add(skill)
                features.append(skill)
    return features
