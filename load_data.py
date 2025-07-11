def load():
    data = []
    labels = []
    
    with open("data.txt", "r") as f:
        for line in f:
            parts = line.strip().split() 
            if len(parts) != 3:
                continue 
            x1, x2, y = map(float, parts)
            data.append([x1, x2])
            labels.append(int(y)) 
    
    return data, labels

data, labels = load()