import heapq

# Node class for Huffman Tree
class Node:
    def __init__(self, char=None, freq=None):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

# Build Huffman Tree
def build_huffman_tree(char_freq):
    heap = [Node(char, freq) for char, freq in char_freq.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(freq=left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    
    return heap[0]

# Generate Huffman Codes
def generate_codes(node, current_code="", codes=None):
    if codes is None:
        codes = {}
    if node is None:
        return codes
    if node.char is not None:
        codes[node.char] = current_code
    generate_codes(node.left, current_code + "0", codes)
    generate_codes(node.right, current_code + "1", codes)
    return codes

# Encode the input string
def encode_string(data, codes):
    return ''.join([codes[char] for char in data])

# Main program
if __name__ == "__main__":
    # Step 1: Input characters and frequencies
    n = int(input("Enter number of characters: "))
    char_freq = {}
    for _ in range(n):
        char = input("Enter character: ")
        freq = int(input(f"Enter frequency for '{char}': "))
        char_freq[char] = freq

    # Step 2: Input the data string to encode
    data = input("Enter string to encode: ")

    # Step 3: Build Huffman Tree
    root = build_huffman_tree(char_freq)

    # Step 4: Generate Huffman Codes
    codes = generate_codes(root)

    # Step 5: Display Huffman Codes
    print("\nHuffman Codes:")
    for char in sorted(codes):
        print(f"{char}: {codes[char]}", end=' ')
    print()

    # Step 6: Encoded output
    encoded = encode_string(data, codes)
    print(f"\nEncoded string: {encoded}")

    # Step 7: Character-wise encoding
    print("\nCharacter-wise Huffman Encoding:")
    for char in data:
        print(f"{char}: {codes[char]}")