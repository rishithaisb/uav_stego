class ArithmeticEncoder:
    def __init__(self):
        self.low = 0.0
        self.high = 1.0
        self.probabilities = {}

    def set_probabilities(self, text):
        total_chars = len(text)
        self.probabilities = {}
        for char in text:
            self.probabilities[char] = self.probabilities.get(char, 0) + 1
        for char in self.probabilities:
            self.probabilities[char] /= total_chars

    def encode(self, text):
        self.set_probabilities(text)
        for char in text:
            range_size = self.high - self.low
            self.high = self.low + range_size * sum(self.probabilities[c] for c in self.probabilities if c <= char)
            self.low = self.low + range_size * sum(self.probabilities[c] for c in self.probabilities if c < char)

        encoded_value = (self.low + self.high) / 2  # Keep the encoded float value
        return encoded_value, self.probabilities  # Return a tuple (float, dict)


class ArithmeticDecoder:
    def __init__(self):
        self.probabilities = {}

def decode(self, encoded_value, text_length, probabilities):
    self.probabilities = probabilities
    decoded_text = ""
    low, high = 0.0, 1.0

    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)  # Ensure consistent order

    for _ in range(text_length):
        range_size = high - low
        cumulative_prob = low

        for char, prob in sorted_probs:
            next_range = cumulative_prob + range_size * prob
            if cumulative_prob <= encoded_value < next_range:
                decoded_text += char
                high = next_range
                low = cumulative_prob
                break
            cumulative_prob = next_range

    return decoded_text
