def load_pa():
    weight = []
    bias = []
    
    with open("parameter.txt", "r") as f:
        for line in f:
            parts = line.strip().split() 
            if len(parts) != 5:
                continue 
            w1, w2, w3, b1, b2 = map(float, parts)
            weight.append([w1, w2, w3])
            bias.append([int(b1), int(b2)]) 
    
    return weight, bias

weight,bias = load_pa()