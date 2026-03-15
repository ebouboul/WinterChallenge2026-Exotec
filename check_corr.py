import csv
import math

def main():
    with open("dataset.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [[float(v) for v in row] for row in reader if len(row) == 6]
        
    for i, name in enumerate(header[:-1]):
        xs = [row[i] for row in data]
        ys = [row[-1] for row in data]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)
        
        corr = cov / math.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else 0
        print(f"Correlation of {name} with label: {corr:.4f}")
        
    bot_diffs = [row[1] for row in data]
    labels = [row[-1] for row in data]
    
    pos_bot_diff_labels = [l for b, l in zip(bot_diffs, labels) if b > 0]
    neg_bot_diff_labels = [l for b, l in zip(bot_diffs, labels) if b < 0]
    
    print(f"Mean label when bot_diff > 0: {sum(pos_bot_diff_labels)/len(pos_bot_diff_labels):.4f} (count: {len(pos_bot_diff_labels)})" if pos_bot_diff_labels else "No bot_diff > 0")
    print(f"Mean label when bot_diff < 0: {sum(neg_bot_diff_labels)/len(neg_bot_diff_labels):.4f} (count: {len(neg_bot_diff_labels)})" if neg_bot_diff_labels else "No bot_diff < 0")

if __name__ == "__main__":
    main()
