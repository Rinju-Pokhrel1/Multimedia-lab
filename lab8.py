def run_length_encoding_fixed_width(input_string):
    if not input_string:
        return ""

    encoded = []
    prev_char = input_string[0]
    count = 1

    for char in input_string[1:]:
        if char == prev_char:
            count += 1
        else:
            encoded.append(f"{prev_char}{count:02d}")
            prev_char = char
            count = 1

    encoded.append(f"{prev_char}{count:02d}")

    return ''.join(encoded)


# --- Main Program ---
# Get input from user
user_input = input("Enter a string to encode: ")

# Run-Length Encode the user input
encoded_rle = run_length_encoding_fixed_width(user_input)

print("Run-Length Encoded String:", encoded_rle)