import csv

def read_csv_file (file_name):
    with open(file_name, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader ]


a = read_csv_file("./data/train_stories.csv")
print (a)