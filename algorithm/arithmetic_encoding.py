class ArithmeticCoding:
    def __init__(self):
        self.prob_table = {}
        self.sorted_chars = [] 

    def build_probability_table(self, text):
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        total = len(text)
        self.prob_table = {char: freq[char] / total for char in freq}
        self.sorted_chars = sorted(self.prob_table.keys())  

    def encode(self, text):
        self.build_probability_table(text)
        low, high = 0.0, 1.0
        
        for char in text:
            range_size = high - low
            char_index = self.sorted_chars.index(char)
            low = low + range_size * sum(self.prob_table[c] for c in self.sorted_chars[:char_index])
            high = low + range_size * self.prob_table[char]
        
        return (low + high) / 2  

    def decode(self, value, length):
        text = ""
        for _ in range(length):
            for char in self.sorted_chars:
                low = sum(self.prob_table[c] for c in self.sorted_chars if c < char)
                high = low + self.prob_table[char]
                if low <= value < high:
                    text += char
                    value = (value - low) / (high - low)
                    break
        return text